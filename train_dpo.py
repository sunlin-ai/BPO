## reference: https://www.philschmid.de/rl-with-llms-in-2025-dpo

import os
import torch
import logging
from datetime import datetime
import torch.nn.functional as F
from datasets import load_dataset
from dataclasses import dataclass
from distutils.util import strtobool
from trl.trainer.utils import cap_exp, pad, pad_to_length
from transformers.trainer_utils import get_last_checkpoint
from trl.trainer.dpo_config import FDivergenceType, FDivergenceConstants
from trl import DPOTrainer, DPOConfig, TrlParser, get_peft_config, ModelConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig

system_prompt = """<|im_start|>system
Please reason step by step, and put your final answer within \\boxed.<|im_end|>
<|im_start|>user
{question} Let's think step by step and output the final answer within \\boxed.<|im_end|>
<|im_start|>assistant
"""


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)


########################
# Helper functions
########################

def get_checkpoint(training_args: DPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


class CustomDPOTrainer(DPOTrainer):
    def dpo_loss(
            self,
            chosen_logps: torch.FloatTensor,
            rejected_logps: torch.FloatTensor,
            ref_chosen_logps: torch.FloatTensor,
            ref_rejected_logps: torch.FloatTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the model for the chosen responses. Shape: `(batch_size,)`.
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the model for the rejected responses. Shape: `(batch_size,)`.
            ref_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: `(batch_size,)`.
            ref_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: `(batch_size,)`.

        Returns:
            A tuple of three tensors: `(losses, chosen_rewards, rejected_rewards)`.
            The losses tensor contains the DPO loss for each example in the batch.
            The `chosen_rewards` and `rejected_rewards` tensors contain the rewards for the chosen and rejected
            responses, respectively.
        """

        balance_factor = 0.3

        device = self.accelerator.device

        # Get the log ratios for the chosen and rejected responses
        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            # The alpha-divergence formula: (1 - u^-alpha) / alpha
            # The divergence difference between the chosen and rejected sample is:
            #     (1 - u[w]^-alpha) / alpha - (1 - u[l]^-alpha) / alpha
            #        = (u[l]^-alpha - u[w]^-alpha) / alpha
            # where u[w] and u[l] are the policy/reference probability ratios
            # for the chosen and rejected samples, respectively.
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps

            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                # The js-divergence formula: log(2 * u / (1 + u))
                # The divergence difference between the chosen and rejected sample is:
                #     log(2 * u[w] / (1 + u[w])) - log(2 * u[l] / (1 + u[l]))
                #       = log(u[w]) - log(u[l]) - (log(1 + u[w]) - log(1 + u[l]))
                # where u[w] and u[l] are the policy/reference probability ratios
                # for the chosen and rejected samples, respectively.
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the
        # labels and calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        elif self.loss_type == "robust":
            losses = (
                             -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                             + F.logsigmoid(-self.beta * logits) * self.label_smoothing
                     ) / (1 - 2 * self.label_smoothing)

        elif self.loss_type == "exo_pair":
            # eqn (16) of the EXO paper: https://huggingface.co/papers/2402.00856
            import math

            if self.label_smoothing == 0:
                self.label_smoothing = 1e-3
            losses = (self.beta * logits).sigmoid() * (
                    F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)
            ) + (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))

        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)

        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2

        elif self.loss_type == "bco_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean
            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )

        elif self.loss_type == "sppo_hard":
            # In the paper (https://huggingface.co/papers/2405.00675), SPPO employs a soft probability approach,
            # estimated using the PairRM score. The probability calculation is conducted outside of the trainer class.
            # The version described here is the hard probability version, where P in Equation (4.7) of Algorithm 1 is
            # set to 1 for the winner and 0 for the loser.
            a = chosen_logps - ref_chosen_logps
            b = rejected_logps - ref_rejected_logps
            losses = (a - 0.5 / self.beta) ** 2 + (b + 0.5 / self.beta) ** 2

        elif self.loss_type == "nca_pair":
            chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
            rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
            losses = (
                    -F.logsigmoid(chosen_rewards)
                    - 0.5 * F.logsigmoid(-chosen_rewards)
                    - 0.5 * F.logsigmoid(-rejected_rewards)
            )

        elif self.loss_type == "aot_pair":
            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            losses = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "aot":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim=0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim=0)
            delta = logratios_sorted - ref_logratios_sorted
            losses = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )

        elif self.loss_type == "apo_zero":
            # Eqn (7) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are better than your model's default output
            losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)  # Increase chosen likelihood
            losses_rejected = F.sigmoid(self.beta * rejected_logratios)  # Decrease rejected likelihood
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "apo_down":
            # Eqn (8) of the APO paper (https://huggingface.co/papers/2408.06266)
            # Use this loss when you believe the chosen outputs are worse than your model's default output.
            # Decrease chosen likelihood and decrease rejected likelihood more
            losses_chosen = F.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            losses = losses_chosen + losses_rejected

        elif self.loss_type == "discopop":
            # Eqn (5) of the DiscoPOP paper (https://huggingface.co/papers/2406.08414)
            # This loss was discovered with LLM discovery
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = logratios - ref_logratios
            logits = logits * self.beta
            # Modulate the mixing coefficient based on the log ratio magnitudes
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            # Blend between logistic and exponential component based on log ratio modulation
            losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation

        elif self.loss_type == "DPO":
            chosen_rewards = (chosen_logps - ref_chosen_logps)
            rejected_rewards = (rejected_logps - ref_rejected_logps)
            losses = -F.logsigmoid(self.beta * (chosen_rewards - rejected_rewards))

        elif self.loss_type == "DDPO":
            lamda = 50
            chosen_rewards = (chosen_logps - ref_chosen_logps)
            rejected_rewards = (rejected_logps - ref_rejected_logps)
            losses = -F.logsigmoid(
                self.beta * (chosen_rewards - rejected_rewards - lamda * torch.relu(ref_chosen_logps - chosen_logps)))

        elif self.loss_type == "Cal-DPO":
            chosen_rewards = (chosen_logps - ref_chosen_logps)
            rejected_rewards = (rejected_logps - ref_rejected_logps)
            DPO_loss = -F.logsigmoid((chosen_rewards - rejected_rewards))
            Cal_loss = F.mse_loss(chosen_rewards,
                                  torch.tensor(1.0 / (2.0 * self.beta)).to(chosen_rewards)) + F.mse_loss(
                rejected_rewards, torch.tensor(-1.0 / (2.0 * self.beta)).to(rejected_rewards))
            losses = DPO_loss + 0.5 * Cal_loss

        elif self.loss_type == "balance_logistic":
            logits = torch.min(chosen_logps - ref_chosen_logps, ref_rejected_logps - rejected_logps)
            logits = self.beta * logits
            losses = -F.logsigmoid(logits)

        elif self.loss_type == "balance_hinge":
            logits = torch.min(chosen_logps - ref_chosen_logps, ref_rejected_logps - rejected_logps)
            logits = self.beta * logits
            losses = torch.relu(1 - logits)

        elif self.loss_type == "balance_square_minus":
            logits = torch.min(chosen_logps - ref_chosen_logps, ref_rejected_logps - rejected_logps)
            logits = self.beta * logits
            losses = (logits - 1) ** 2

        elif self.loss_type == "balance_exponential":
            logits = torch.min(chosen_logps - ref_chosen_logps, ref_rejected_logps - rejected_logps)
            logits = self.beta * logits
            losses = torch.exp(-logits)

        elif self.loss_type == "balance_truncated_quadratic":
            logits = torch.min(chosen_logps - ref_chosen_logps, ref_rejected_logps - rejected_logps)
            logits = self.beta * logits
            losses = (torch.relu(1 - logits)) ** 2

        elif self.loss_type == "balance_savage":
            logits = torch.min(chosen_logps - ref_chosen_logps, ref_rejected_logps - rejected_logps)
            logits = self.beta * logits
            losses = 1 / (1 + torch.exp(logits)) ** 2

        elif self.loss_type == "balance_logistic_alpha":
            logits = torch.min(chosen_logps - ref_chosen_logps, balance_factor * (ref_rejected_logps - rejected_logps))
            logits = self.beta * logits
            losses = -F.logsigmoid(logits)

        elif self.loss_type == "balance_hinge_alpha":
            logits = torch.min(chosen_logps - ref_chosen_logps, balance_factor * (ref_rejected_logps - rejected_logps))
            logits = self.beta * logits
            losses = torch.relu(1 - logits)

        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'exo_pair', "
                "'nca_pair', 'robust', 'bco_pair', 'sppo_hard', 'aot', 'aot_pair', 'discopop', 'apo_zero', 'apo_down']"
            )

        chosen_rewards = self.beta * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = self.beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

        return losses, chosen_rewards, rejected_rewards


def dpo_function(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: DPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ###############
    # Load datasets
    ###############
    if script_args.dataset_id_or_path.endswith(".json"):
        train_dataset = load_dataset(
            "json", data_files=script_args.dataset_id_or_path, split="train"
        )
    else:
        train_dataset = load_dataset(
            script_args.dataset_id_or_path, split=script_args.dataset_splits
        )

    logger.info(
        f"Loaded dataset with {len(train_dataset)} samples and the following features: {train_dataset.features}"
    )

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    #####################
    # Prepare and format dataset
    #####################
    def format_dpo_sample(sample):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sample["system_prompt"]},
                {"role": "user", "content": sample["prompt"]},
            ],
            tokenize=False,
        )
        chosen = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": sample["chosen"]}], tokenize=False
        )
        rejected = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": sample["rejected"]}], tokenize=False
        )
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    def format_dpo_sample_orz(sample):
        # prompt = tokenizer.apply_chat_template(
        #     [
        #         {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        #         {"role": "user", "content": str(sample["question"])+" Let's think step by step and output the final answer within \\boxed{}."},
        #     ],
        #     tokenize=False,
        # )
        # chosen = tokenizer.apply_chat_template(
        #     [{"role": "assistant", "content": sample["chosen"]["solution"]}], tokenize=False
        # )
        # rejected = tokenizer.apply_chat_template(
        #     [{"role": "assistant", "content": sample["rejected"]["solution"]}], tokenize=False
        # )

        prompt = system_prompt.format(question=sample["question"])
        prompt = prompt.replace("\\boxed", "\\boxed{}")
        chosen = sample["chosen"]["solution"]
        rejected = sample["rejected"]["solution"]

        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
    format_fn = format_dpo_sample_orz if "orz_math" in script_args.dataset_id_or_path else format_dpo_sample

    train_dataset = train_dataset.map(format_fn, remove_columns=train_dataset.column_names)

    # remove all columns except chosen, rejected
    print(f"Columns: {train_dataset.features.keys()}")
    train_dataset = train_dataset.select_columns(["prompt", "chosen", "rejected"])

    #######################################
    # Load the model and/or reference model
    #######################################
    model_kwargs = dict(
        revision=model_args.model_revision,  # What revision from Huggingface to use, defaults to main
        trust_remote_code=model_args.trust_remote_code,
        # Whether to trust the remote code, this also you to fine-tune custom architectures
        attn_implementation=model_args.attn_implementation,
        # What attention implementation to use, defaults to flash_attention_2
        torch_dtype=(
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        ),  # What torch dtype to use, defaults to auto
        use_cache=False if training_args.gradient_checkpointing else True,  # Whether
        low_cpu_mem_usage=(
            True
            if not strtobool(os.environ.get("ACCELERATE_USE_DEEPSPEED", "false"))
            else None
        ),  # Reduces memory usage on CPU for loading the model
    )

    # Check which training method to use and if 4-bit quantization is needed
    if model_args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
            bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
        )
    if model_args.use_peft:
        peft_config = get_peft_config(model_args)
    else:
        peft_config = None

    # Policy Model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    # Checks wether we use adapters for reference model or not
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    else:
        model_ref = None

    #########################
    # Instantiate DPO trainer
    #########################
    trainer = CustomDPOTrainer(
        model,
        ref_model=model_ref,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    if trainer.is_fsdp_enabled and peft_config:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["sft", "tutorial", "philschmid"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, DPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Run the main training loop
    dpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
