import argparse
import gc
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import wandb
from datasets import load_dataset
from deepspeed import DeepSpeedEngine
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

from utils import (
    clean_up_checkpoints,
    close_to_zero,
    compute_token_log_probs,
    dump_episodes,
    evaluate_on_test_set,
    find_last_checkpoint,
    initialize_training_process_group,
    load_model_into_vllm,
    prepare_model_inputs,
)

os.environ["VLLM_USE_V1"] = "0"

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


# Load and process dataset
def preprocess_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    PROMPT_TEMPLATE: str,
):
    numbers: List[int] = example["nums"]
    target: int = example["target"]
    prompt = PROMPT_TEMPLATE.format(numbers=numbers, target=target)
    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return {"prompt": prompt, "input_ids": input_ids}


def format_correct_func(completion: str, EOS_TOKEN: str) -> float:
    """
    Format: Reasoning process <answer> answer here <answer>

    Also checks that the content within <answer>...</answer> conforms to a
    specified pattern (only digits, + - * / ( ) . and whitespace).

    Args:
        completion (str): Generated output
        EOS_TOKEN (str): End of sequence token

    Returns:
        float: Reward score
    """
    # Define the allowed pattern (only numbers, +, -, *, /, (, ), ., and whitespace)
    allowed_pattern = r"^[\d+\-*/().\s]+$"

    try:
        # Check if the format is correct
        # Pattern means:
        # 3) <answer>...anything...</answer>
        regex = r"<answer>(.*?)<\/answer>"
        match = re.search(regex, completion, re.DOTALL)

        if match is None or len(match.groups()) != 1:
            # Format is incorrect
            return 0.0
        else:
            # Extract the content inside <answer>...</answer>
            answer_content = match.group(1).strip()

            # Check if answer content matches the allowed pattern
            if not re.match(allowed_pattern, answer_content):
                # If it doesn't match, reward is 0.5
                return 0.5
            else:
                # If both format and pattern are correct, reward is 1
                return 1.0
    except Exception:
        # Any error leads to 0 reward
        return 0.0


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """
    Evaluates completion based on mathematical correctness of the answer

    Args:
        completion (str): Generated output
        target (str): Expected answer
        nums (list): Available numbers to use in the equation

    Returns:
        float: Reward score
    """
    try:
        # Check if the format is correct
        match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
        if match is None:
            return 0.0
        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation):
            return 0.0

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return 1.0
        else:
            return 0.0
    except Exception:
        # If evaluation fails, reward is 0
        return 0.0


def compute_reward(completion: str, sample: Dict[str, Any], EOS_TOKEN: str) -> Tuple[float, Dict[str, float]]:
    nums = sample["nums"]
    target = sample["target"]

    format_correct = format_correct_func(completion, EOS_TOKEN)
    equation_reward = equation_reward_func(completion=completion, nums=nums, target=target)

    reward = equation_reward

    metrics = {
        "format_correct": format_correct,
        "equation_reward": equation_reward,
    }

    return reward, metrics


def create_training_episodes(
    samples: List[Dict[str, Any]] = None,
    all_generations: List[List[int]] = None,
    all_finish_reasons: List[str] = None,
    tokenizer: AutoTokenizer = None,
    EOS_TOKEN: str = None,
    GENERATIONS_PER_SAMPLE: int = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process model generations and calculate rewards for training episodes.

    This function processes generated responses and calculates rewards for training episodes by:
    1. Grouping generations by sample (GENERATIONS_PER_SAMPLE responses per input)
    2. Computing rewards and advantages for each response
    3. Processing response tokens (adding EOS tokens where needed)

    Args:
        samples: List of input samples, each containing:
            - input_ids: List[int], tokenized input prompt
            - nums: List[int], numbers to use in equation
            - target: int, target value for equation
        all_generations: List of token ID sequences for each generated response
        all_finish_reasons: List of finish reasons for each generation ("stop" or other)

    Returns:
        Tuple containing:
        1. Dictionary with processed data for training:
            - all_query_token_ids: List[List[int]], input token IDs repeated for each generation
            - all_response_token_ids: List[List[int]], response token IDs with EOS tokens added
            - all_advantages: List[List[float]], advantage values repeated for each token
        2. Dictionary with generation statistics:
            - response_lengths: List[int], lengths of generated responses
            - rewards: List[float], raw reward values
            - non_stop_rate: List[bool], whether each generation ended naturally
            - reward_metrics/*: Various reward component metrics

    Example:
        >>> samples = [{"input_ids": [1,2,3], "nums": [1,2,3], "target": 6}]
        >>> generations = [[4,5], [6,7], [8,9]]  # 3 generations per sample
        >>> finish_reasons = ["stop", "length", "stop"]
        >>> episodes, stats = create_training_episodes(samples, generations, finish_reasons)
        >>> episodes
        {
            'all_query_token_ids': [[1,2,3], [1,2,3], [1,2,3]],
            'all_response_token_ids': [[4,5,EOS], [6,7], [8,9,EOS]],
            'all_advantages': [[0.5,0.5,0.5], [-1.0,-1.0], [0.5,0.5,0.5]]
        }
    """
    assert len(all_generations) == len(all_finish_reasons)
    assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE

    # Process responses and calculate rewards
    groups = [
        list(range(i, i + GENERATIONS_PER_SAMPLE)) for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
    ]  # example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

    stats = {
        "response_lengths": [],
        "response_lengths_correct": [],
        "response_lengths_wrong": [],
        "rewards": [],
        "non_stop_rate": [],
        "pass_at_group": [],
    }

    for sample, group_indices in zip(samples, groups):
        response_token_ids = [all_generations[i] for i in group_indices]
        finish_reasons = [all_finish_reasons[i] for i in group_indices]
        responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
        rewards_and_metrics = [compute_reward(resp, sample, EOS_TOKEN) for resp in responses]
        rewards, reward_metrics = zip(*rewards_and_metrics)

        rewards = np.array(rewards)
        # GRPO: subtract the group mean reward and divide by std deviation
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        per_token_advantages = [[adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)]

        all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
        all_responses_token_ids.extend(response_token_ids)
        all_advantages.extend(per_token_advantages)

        stats["pass_at_group"].append(0 if max(rewards) < 1 else 1)
        stats["rewards"].extend(rewards)
        stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        response_lengths = [len(ids) for ids in response_token_ids]
        stats["response_lengths"].extend(response_lengths)
        stats["response_lengths_correct"].extend(
            [length for length, reward in zip(response_lengths, rewards) if reward >= 1.0]
        )
        stats["response_lengths_wrong"].extend(
            [length for length, reward in zip(response_lengths, rewards) if reward < 1.0]
        )
        for rm in reward_metrics:
            for k, v in rm.items():
                stats.setdefault(f"reward_metrics/{k}", []).append(v)

    episodes = {
        "all_query_token_ids": all_query_token_ids,
        "all_response_token_ids": all_responses_token_ids,
        "all_advantages": all_advantages,
    }

    return episodes, stats


def compute_pg_loss(
    policy_model: Union[DeepSpeedEngine, PreTrainedModel],
    batch: Dict[str, torch.Tensor],
    total_response_len: torch.Tensor,
    TEMPERATURE: float,
    KL_COEFFICIENT: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the policy gradient loss with KL penalty between policy and reference models.

    This function:
    1. Calculates KL divergence penalty between the models
    2. Computes policy gradient loss using advantages
    3. Combines the losses with KL coefficient

    Args:
        policy_model: The model being trained
        batch: Dictionary containing:
            - input_ids: Tensor of shape [batch_size, seq_len]
            - attention_mask: Tensor of shape [batch_size, seq_len]
            - labels: Tensor of shape [batch_size, seq_len] with -100 for ignored positions
            - advantages: Tensor of shape [batch_size, seq_len]
            - ref_log_probs: Tensor of shape [batch_size, seq_len-1]
        total_response_len: Total number of valid tokens in the batch. This is a scalar tensor.

    Returns:
        Tuple containing:
            - loss: Combined policy gradient and KL penalty loss (scalar tensor)
            - metrics: Dictionary with detailed loss components:
                - policy_loss: Pure policy gradient loss
                - kl_penalty: KL divergence penalty
                - entropy: Policy entropy
    """
    input_ids = batch["input_ids"]  # [batch_size, seq_len]
    attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
    labels = batch["labels"]  # [batch_size, seq_len]
    labels_mask = batch["labels_mask"]  # [batch_size, seq_len]
    advantages = batch["advantages"]  # [batch_size, seq_len]
    ref_logps = batch["ref_log_probs"]  # [batch_size, seq_len-1]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "labels_mask": labels_mask,
    }

    logps = compute_token_log_probs(policy_model, model_inputs, TEMPERATURE)  # [batch_size, seq_len-1]
    labels_mask = labels_mask[..., 1:].to(logps.dtype)  # [batch_size, seq_len-1]

    # KL3 calculation from http://joschu.net/blog/kl-approx.html
    ref_logratio = ref_logps - logps
    kl_penalty = torch.exp(ref_logratio) - 1 - ref_logratio  # [batch_size, seq_len-1]
    kl_penalty = kl_penalty * labels_mask  # [batch_size, seq_len-1]

    with torch.no_grad():
        entropy = -logps.sum() / labels_mask.sum()  # scalar
        zero_advantages = close_to_zero(advantages[..., 1:], labels_mask)  # scalar

    policy_loss = -logps * advantages[..., 1:]
    policy_loss = policy_loss * labels_mask

    # GRPO: divide the loss by the length of the responses
    loss = (policy_loss + KL_COEFFICIENT * kl_penalty).sum() / total_response_len

    metrics = {
        "policy_loss": policy_loss.sum().item() / total_response_len.item(),
        "kl_penalty": kl_penalty.sum().item() / total_response_len.item(),
        "entropy": entropy.item() / total_response_len.item(),
        "zero_advantages_ratio": zero_advantages.item() / total_response_len.item(),
    }

    return loss, metrics


def main(args):
    initialize_training_process_group(0, 1)
    # ignore
    curr_cuda_device = torch.device("cuda")

    ############################################
    # Hyperparameters
    ############################################

    # Model configuration
    MODEL_NAME = args.model_name

    # RL parameters
    # Total number of training iterations
    NUM_ITERATIONS = args.total_episodes // args.episodes_per_step
    # Number of episodes to collect per iteration for training
    EPISODES_PER_ITERATION = args.episodes_per_step
    # Number of responses to generate for each input prompt
    GENERATIONS_PER_SAMPLE = args.num_responses_per_prompt
    # Controls how much the policy can deviate from the reference model
    KL_COEFFICIENT = args.kl_coeff

    # Training hyperparameters
    # Batch size for each GPU device during training
    PER_DEVICE_BATCH_SIZE = args.per_device_batch_size
    # Learning rate for model updates
    LEARNING_RATE = 5e-7

    # Sampling parameters
    # Maximum number of tokens to generate in each response
    MAX_RESPONSE_TOKENS = args.max_response_tokens
    # Controls randomness in generation (higher = more random)
    TEMPERATURE = args.temperature
    # Nucleus sampling parameter (1.0 = disabled)
    TOP_P = 0.999  # to avoid sampling unused tokens absent from tokenizer see https://github.com/vllm-project/vllm/issues/13175#issuecomment-2781842571
    # Top-k sampling parameter (-1 = disabled)
    TOP_K = -1  # no top k
    # DeepSpeed configuration
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2},
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "torch_adam": True,
                "fused": True,
            },
        },
    }
    ref_deepspeed_config = {
        "bf16": {"enabled": True},
        # No effect
        "train_batch_size": EPISODES_PER_ITERATION,
        "train_micro_batch_size_per_gpu": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": EPISODES_PER_ITERATION // PER_DEVICE_BATCH_SIZE,
    }

    model_name_short = MODEL_NAME.split("/")[-1]
    if args.run_id is None:
        RUN_NAME = f"{model_name_short}_temp{TEMPERATURE}_kl{KL_COEFFICIENT}_lr{LEARNING_RATE}"
    else:
        RUN_NAME = args.run_id

    model_name_short = MODEL_NAME.split("/")[-1]
    RUN_NAME = f"{model_name_short}_temp{TEMPERATURE}_kl{KL_COEFFICIENT}_lr{LEARNING_RATE}"
    EXP_DIR = Path(args.output_dir) / RUN_NAME

    if EXP_DIR.exists():
        logging.warning(f"You are running an experiment that will restart from {EXP_DIR}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    ############################################
    # Prompts and Dataset
    ############################################

    PROMPT_TEMPLATE = (
        "Combine a couple numbers in a simple algebraic equation so that it equals a target number. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Think about the reasoning process then return the final equation inside <answer> </answer> tags. "
        "Do not include the = target inside the <answer> </answer> tags. "
        "For example with numbers 5 2 3 to equal target 4, after thinking step by step we see (5 + 2) - 3 = 4 so output <answer>(5 + 2) - 3</answer>.\n\n"
        "Use the numbers {numbers} to create an equation that equals {target}.\n\n"
        "Let's solve this step by step."
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # TODO FIX
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    # Rank 0 will preprocess the dataset first
    dataset = dataset.map(
        preprocess_example,
        num_proc=6,
        fn_kwargs={
            "tokenizer": tokenizer,
            "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        },
        desc="Preprocessing dataset",
    )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=500, seed=args.seed)
    train_dataset = train_test_split["train"]
    orig_train_dataset_size = len(train_dataset)
    test_dataset = train_test_split["test"]

    logger.info(f"Train dataset size: {orig_train_dataset_size}; each rank will process {len(train_dataset)} samples")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    ############################################
    # Initialize Models
    ############################################

    policy_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )

    # reference_model.module.cpu()

    ############################################
    # Initialize vLLM (Inference) engine
    ############################################

    # Disable root vllm logger for non-main ranks
    vllm_logger = logging.getLogger("vllm")
    vllm_logger.setLevel(logging.ERROR)

    inference_engine = LLM(
        model=MODEL_NAME,
        skip_tokenizer_init=False,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
        swap_space=4,
        scheduling_policy="fcfs",
        dtype=torch.bfloat16,
        max_model_len=MAX_RESPONSE_TOKENS + 1024,
        enable_sleep_mode=True,
        device=f"cuda:{torch.cuda.current_device()}",
        tensor_parallel_size=1,
    )

    # Wandb for logging. Only rank 0 will initialize wandb
    wandb.init(
        project="nano-aha-moment",
        name=RUN_NAME,
        resume="allow",
        config=vars(args),
    )

    sampler_rng = np.random.default_rng(seed=args.seed)
    NUM_SAMPLES_PER_ITERATION = EPISODES_PER_ITERATION // GENERATIONS_PER_SAMPLE

    # Load checkpoint if it exists
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None:
        logger.info(f"Resuming from checkpoint {ckpt_path} at iteration {ckpt_iter}")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        begin_iter = ckpt_iter + 1
        load_model_into_vllm(policy_model, inference_engine)

        logger.info(f"Skipping {ckpt_iter} rounds of samples")
        for _ in trange(ckpt_iter):
            _ = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_ITERATION, replace=False)

    for iteration in trange(begin_iter, NUM_ITERATIONS):
        logger.info(f"Iteration {iteration}/{NUM_ITERATIONS}")
        start_time = time.time()

        metrics = {}

        #########################################################
        # Evaluation
        #########################################################
        eval_sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=MAX_RESPONSE_TOKENS,
            n=1,
            detokenize=True,
            stop=["</answer>", EOS_TOKEN],
            include_stop_str_in_output=True,
        )

        eval_stats = None
        if iteration % (NUM_ITERATIONS // args.num_evals) == 0 and iteration > 0:
            logger.info("Evaluating on eval set...")
            eval_episodes, eval_stats = evaluate_on_test_set(
                inference_engine=inference_engine,
                test_dataset=test_dataset,
                tokenizer=tokenizer,
                eval_sampling_params=eval_sampling_params,
                reward_func=lambda completion, sample: compute_reward(completion, sample, EOS_TOKEN),
            )
            eval_episode_table = dump_episodes(
                episodes=eval_episodes,
                episodes_stats=eval_stats,
                exp_dir=EXP_DIR,
                tokenizer=tokenizer,
                iteration=iteration,
                is_eval=True,
            )
            wandb.log({"eval/episodes": eval_episode_table, "iteration": iteration})

        #########################################################
        # Generate Episodes
        #########################################################

        train_sampling_params = SamplingParams(
            n=GENERATIONS_PER_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_tokens=MAX_RESPONSE_TOKENS,
            detokenize=True,
            stop=["</answer>", EOS_TOKEN],
            include_stop_str_in_output=True,
        )

        # Sample training batch
        indices = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_ITERATION, replace=False)
        samples = train_dataset.select(indices)

        gen_time = time.time()

        # Sample responses
        outputs = inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=train_sampling_params,
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]

        logger.info(f"Generated {len(all_generations)} responses")
        logger.info(f"Time taken to generate {len(all_generations)} responses: {time.time() - gen_time} seconds")

        # Process responses and calculate rewards
        episodes, episodes_stats = create_training_episodes(
            samples=samples,
            all_generations=all_generations,
            all_finish_reasons=all_finish_reasons,
            tokenizer=tokenizer,
            EOS_TOKEN=EOS_TOKEN,
            GENERATIONS_PER_SAMPLE=GENERATIONS_PER_SAMPLE,
        )

        inference_engine.sleep(1)
        gc.collect()
        torch.cuda.empty_cache()
        # time.sleep(1)

        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            exp_dir=EXP_DIR,
            tokenizer=tokenizer,
            iteration=iteration,
            do_save=iteration % 10 == 0 or iteration == 0,
        )

        #########################################################
        # Training
        #########################################################

        # Prepare training batch
        model_inputs = prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device=curr_cuda_device,
        )

        # This is old code that frees up more space if you're running 3B models
        # logger.info("Moving reference model to GPU")
        # reference_model.module.to(curr_cuda_device)
        # reference_model.eval()

        with torch.no_grad():
            ref_log_probs = []
            for i in trange(
                0,
                EPISODES_PER_ITERATION,
                PER_DEVICE_BATCH_SIZE,
                desc="Computing reference logprobs",
            ):
                batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}
                ref_log_probs.append(
                    compute_token_log_probs(
                        model=reference_model,
                        inputs=batch,
                        temperature=TEMPERATURE,
                    )
                )
            ref_log_probs = torch.cat(ref_log_probs)
            model_inputs["ref_log_probs"] = ref_log_probs
            del ref_log_probs

        # This is old code that frees up more space if you're running 3B models
        # Free memory taken by reference model
        # logger.info("Moving reference model back to CPU")
        # reference_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        # time.sleep(1)

        # Calculate losses and update model
        policy_model.train()
        total_response_len = (model_inputs["labels"] != -100).sum()
        train_time = time.time()

        for i in trange(
            0,
            EPISODES_PER_ITERATION,
            PER_DEVICE_BATCH_SIZE,
            desc="Gradient Accumulation",
        ):
            batch = {k: v[i : i + PER_DEVICE_BATCH_SIZE] for k, v in model_inputs.items()}

            # Compute policy gradient loss
            loss, loss_metrics = compute_pg_loss(
                policy_model=policy_model,
                batch=batch,
                total_response_len=total_response_len,
                TEMPERATURE=TEMPERATURE,
                KL_COEFFICIENT=KL_COEFFICIENT,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            # Backpropagation and optimization step
            # scale_wrt_gas=False because we are already normalizing by total_response_len
            policy_model.backward(loss, scale_wrt_gas=False)
            del loss, loss_metrics

            policy_model.step()

        logger.info(f"Time taken to train: {time.time() - train_time} seconds")

        #########################################################
        # Update inference engine weights
        #########################################################

        gc.collect()
        torch.cuda.empty_cache()
        # time.sleep(1)

        inference_engine.wake_up()
        load_model_into_vllm(policy_model, inference_engine)

        #########################################################
        # Log metrics
        #########################################################

        train_metrics = {k: np.mean(v) for k, v in metrics.items() if None not in v}
        train_metrics["learning_rate"] = policy_model.get_lr()[0]
        logs = {
            "iteration": iteration,
            f"episodes/iter_{iteration:06d}": episode_table,
            **{f"train/{k}": v for k, v in train_metrics.items()},
        }
        if eval_stats is not None:
            logs.update({f"eval/{k}": np.mean(v) for k, v in eval_stats.items()})
        wandb.log(logs)

        selected_keys = [
            "train/kl_penalty",
            "train/rewards",
            "train/reward_metrics/format_correct",
            "train/reward_metrics/equation_reward",
            "train/pass_at_group",
            "train/response_lengths",
            "eval/rewards",
            "eval/reward_metrics/format_correct",
            "eval/reward_metrics/equation_reward",
        ]
        selected_metrics = {k: float(logs[k]) for k in selected_keys if k in logs}
        logger.info(f"KEY METRICS: {selected_metrics}")
        logger.info(f"Total per iteration {time.time() - start_time} seconds")

    if args.do_save:
        ckpt_dir = EXP_DIR / "checkpoints" / f"ckpt_{iteration:06d}"

        logger.info("Saving HF model")
        policy_model.module.save_pretrained(str(ckpt_dir / "hf_model"))
        tokenizer.save_pretrained(str(ckpt_dir / "hf_model"))

        logger.info("Saving DeepSpeed checkpoint")
        policy_model.save_checkpoint(str(ckpt_dir / "deepspeed"))

        clean_up_checkpoints(
            exp_dir=EXP_DIR,
            keep_every_n_steps=50,
            exclude=[ckpt_dir],
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    # Parse command line arguments
    arg_parser = argparse.ArgumentParser(description="Train small model on countdown with GRPO")
    arg_parser.add_argument("--seed", type=int, default=42, help="A number that initializes our ")
    arg_parser.add_argument("--kl_coeff", type=float, default=0.001, help="KL coefficient for GRPO")
    arg_parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    arg_parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B-base", help="Model name/path")
    arg_parser.add_argument("--per_device_batch_size", type=int, default=16, help="Per device batch size")
    arg_parser.add_argument("--max_response_tokens", type=int, default=1024, help="Max response tokens")
    arg_parser.add_argument("--learning_rate", type=float, default=5e-7, help="Learning rate for training")
    arg_parser.add_argument("--num_evals", type=int, default=5, help="How many evals to do over the training")
    arg_parser.add_argument(
        "--total_episodes", type=int, default=6400, help="Total number of prompt-completions to generate per update"
    )
    arg_parser.add_argument(
        "--episodes_per_step", type=int, default=128, help="Total number of prompt-completions to generate per update"
    )
    arg_parser.add_argument(
        "--num_responses_per_prompt",
        type=int,
        default=8,
        help="Number of completions to generate for each prompt, that becomes the size of the GRPO group",
    )
    arg_parser.add_argument("--run_id", type=str, default=None, help="Run ID")
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to output checkpoints etc",
    )
    arg_parser.add_argument(
        "--do_save",
        action="store_true",
        help="Save final checkpoints to output_dir",
    )
    args = arg_parser.parse_args()

    main(args)
