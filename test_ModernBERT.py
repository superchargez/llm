# from transformers import AutoTokenizer, AutoModelForMaskedLM
# import torch
# def dummy_compile(*args, **kwargs):
#     def decorator(fn):
#         return fn
#     return decorator

# torch.compile = dummy_compile

# model_id = "answerdotai/ModernBERT-large"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForMaskedLM.from_pretrained(model_id)

# text = "The capital of France is [MASK]."
# inputs = tokenizer(text, return_tensors="pt")
# outputs = model(**inputs)

# # To get predictions for the mask:
# masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
# predicted_token_id = outputs.logits[0, masked_index].argmax(axis=-1)
# # predicted_token = tokenizer.decode(predicted_token_id)
# predicted_token = tokenizer.decode([predicted_token_id])#[0]
# print("Predicted token:", predicted_token)
# # Predicted token:  Paris
