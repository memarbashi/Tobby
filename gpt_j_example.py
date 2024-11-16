from transformers import AutoTokenizer, AutoModelForCausalLM

# بارگذاری مدل و توکنایزر GPT-J
model_name = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# تابعی برای تولید متن
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=150)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# مثال استفاده
prompt = "سلام، امروز می‌خواهم در مورد هوش مصنوعی صحبت کنم."
generated_text = generate_text(prompt)
print(generated_text)
