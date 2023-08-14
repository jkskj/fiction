import os

import torch
from flask import Flask, request, jsonify
from transformers import BertTokenizerFast, BertForMaskedLM

app = Flask(__name__)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')
if os.path.exists(os.path.join('fiction-bert-base-chinese', 'bert_fiction_model.bin')):
    state = torch.load(os.path.join('fiction-bert-base-chinese', 'bert_fiction_model.bin'))
    model.load_state_dict(state)
    print("## 成功载入已有模型进行推理......")
else:
    raise ValueError(f"模型 bert_fiction_model.bin 不存在！")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()


def compute_attention_mask(segment_ids):
    idxs = torch.cumsum(segment_ids, dim=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    return mask


@app.route('/fiction', methods=['GET'])
@torch.no_grad()
def predict():
    sen = request.args.get('text')
    print(sen)
    num = 300
    sen1 = ""
    for i in range(num):
        token = tokenizer(sen + '[EOS]', sen1 + "[MASK]", add_special_tokens=False, return_tensors='pt',truncation=True, max_length=512)
        input = token['input_ids'].to(device)
        token_type_ids = token["token_type_ids"]
        attention_mask = compute_attention_mask(token_type_ids)
        attention_mask = attention_mask.to(device)
        output = model(input_ids=input, attention_mask=attention_mask).logits
        pred = torch.argmax(output, dim=-1)
        pred = pred.data.cpu().numpy().tolist()[0][-1]
        pred = tokenizer.decode(pred)
        sen1 = sen1 + pred
        if pred == '[EOS]':
            print("预测结束")
            break
    print(sen1)
    return jsonify(code=200, msg="success", data={'text': sen1})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port="80")
