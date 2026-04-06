import argparse
import sys
from .model_manager import ModelManager
from .server import app
import uvicorn


def _chat_loop(manager, args):
    quant = getattr(args, "quant", None)
    print(f"Loading model {args.model} (quant={quant or 'auto'})...")
    model, tokenizer = manager.load(args.model, quant=quant)
    print(f"Model loaded. Type 'quit' or 'exit' to leave, 'clear' to reset history.\n")

    messages = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": args.system}]
            print("Conversation cleared.\n")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(model.device)

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature if args.temperature > 0 else None,
            "do_sample": args.temperature > 0,
        }
        if args.temperature <= 0:
            generate_kwargs.pop("temperature")

        outputs = model.generate(**generate_kwargs)
        response = tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"Assistant: {response}\n")
        messages.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(prog="cas", description="caS - catch AI models")
    subparsers = parser.add_subparsers(dest="command")

    pull_parser = subparsers.add_parser("pull", help="Download a model")
    pull_parser.add_argument("model", help="Model ID (e.g., meta-llama/Llama-2-7b-hf)")
    pull_parser.add_argument(
        "--source",
        choices=["huggingface", "hf-mirror", "modelscope"],
        default="huggingface",
        help="Model source registry",
    )
    pull_parser.add_argument(
        "--gguf",
        action="store_true",
        help="Download GGUF format only (saves disk space)",
    )
    pull_parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="Specific GGUF quantization (e.g., Q4_K_M, Q8_0, Q5_K_M)",
    )

    list_parser = subparsers.add_parser("list", help="List local models")
    list_parser.add_argument(
        "--verbose", action="store_true", help="Show model details"
    )

    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument(
        "--source",
        choices=["huggingface", "hf-mirror", "modelscope"],
        default="huggingface",
        help="Default model source",
    )

    run_parser = subparsers.add_parser("run", help="Run inference")
    run_parser.add_argument("model", help="Model ID")
    run_parser.add_argument("prompt", help="Prompt text")
    run_parser.add_argument("--max-tokens", type=int, default=100)
    run_parser.add_argument("--temperature", type=float, default=0.7)
    run_parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="GGUF quantization to use (e.g., Q4_K_M)",
    )

    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.add_argument("model", help="Model ID")
    chat_parser.add_argument("--max-tokens", type=int, default=512)
    chat_parser.add_argument("--temperature", type=float, default=0.7)
    chat_parser.add_argument(
        "--system", type=str, default="You are a helpful assistant."
    )
    chat_parser.add_argument(
        "--quant",
        type=str,
        default=None,
        help="GGUF quantization to use (e.g., Q4_K_M)",
    )

    args = parser.parse_args()
    manager = ModelManager(source=getattr(args, "source", "huggingface"))

    if args.command == "pull":
        manager.pull(
            args.model,
            source=args.source,
            gguf=getattr(args, "gguf", False),
            quant=getattr(args, "quant", None),
        )
    elif args.command == "list":
        for m in manager.list_models():
            info = manager.models[m]
            if args.verbose:
                mtype = info.get("type", "unknown")
                quant = info.get("quant", "")
                quant_str = f" ({quant})" if quant else ""
                print(
                    f"{m}  [{info.get('source', 'unknown')}] ({mtype}{quant_str}) -> {info['path']}"
                )
            else:
                print(m)
    elif args.command == "serve":
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.command == "run":
        model, tokenizer = manager.load(args.model, quant=getattr(args, "quant", None))
        inputs = tokenizer(args.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None and hasattr(attention_mask, "to"):
            attention_mask = attention_mask.to(model.device)
        generate_kwargs = {"input_ids": input_ids, "max_new_tokens": args.max_tokens}
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        if args.temperature > 0:
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["do_sample"] = True
        outputs = model.generate(**generate_kwargs)
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    elif args.command == "chat":
        _chat_loop(manager, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
