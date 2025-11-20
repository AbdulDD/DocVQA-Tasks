def preprocess_function_training(dataset):
    images = dataset["image_raw"]
    questions = dataset["question"]
    answers = dataset["answers"]

    batch_inputs = {"input_ids": [], "attention_mask": [], "pixel_values": [], "labels": []}

    for image, question, answer in zip(images, questions, answers):

        # Ensure answer is a string
        if isinstance(answer, list):
            answer = answer[0]

        # 1. Build training prompt
        prompt = (
            "USER: <image>\n"
            f"Question: {question}\n"
            "ASSISTANT:"
        )

        # 2. Tokenize prompt + image
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,         # encoder side
        )

        input_ids = inputs["input_ids"]               # [1,1024]
        attention_mask = inputs["attention_mask"]     # [1,1024]
        pixel_values = inputs["pixel_values"]         # [1,3,336,336]

        # 3. Tokenize answer
        answer_ids = processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )["input_ids"]                                # [1,512]

        # 4. Create labels aligned with input_ids
        labels = torch.full_like(input_ids, -100)     # [1,1024]

        # number of tokens in prompt (excluding padding)
        prompt_len = attention_mask.sum().item()

        # place answer tokens *after* prompt tokens
        end_pos = prompt_len + answer_ids.shape[1]
        if end_pos > labels.shape[1]:
            end_pos = labels.shape[1]

        labels[:, prompt_len:end_pos] = answer_ids[:, : (end_pos - prompt_len)]

        # 5. Append to batch lists
        batch_inputs["input_ids"].append(input_ids[0])
        batch_inputs["attention_mask"].append(attention_mask[0])
        batch_inputs["pixel_values"].append(pixel_values[0])
        batch_inputs["labels"].append(labels[0])

        # Debug Prints
        print(f"input ids shape: {input_ids.shape}")
        print(f"attention mask shape: {attention_mask.shape}")
        print(f"pixel values shape: {pixel_values.shape}")
        print(f"label ids shape: {answer_ids.shape}")
        print(f"labels shape: {labels.shape}")
        print("-" * 50)

    # 6. Convert lists into tensors
    for k in batch_inputs:
        batch_inputs[k] = torch.stack(batch_inputs[k])

    return batch_inputs
