import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizerFast

from apex import amp
from pypinyin import pinyin, Style


class Config:
    def __init__(self, from_path, batch_size, epochs, learning_rate, weight_decay, device, mlm_probability,
                 save_path):
        self.save_path = save_path
        self.from_path = from_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.end_token = '[EOS]'
        self.mlm_probability = mlm_probability


def cache(func):
    """
    本修饰器的作用是将数据预处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        data_path = os.path.join("../", f"data/data.pt")
        if not os.path.exists(data_path):
            if not os.path.exists("../data"):
                os.makedirs(f'../data')
            print(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            print(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class TrainDataset(Dataset):

    def __init__(self, filepathList, tokenizer, cofig):
        self.tokenizer = tokenizer
        self.config = cofig
        data = self.make_data(
            filepathList=filepathList)
        self.sentences1 = data["sentences1"]
        self.sentences2 = data["sentences2"]

    def __len__(self):
        return min(len(self.sentences1), len(self.sentences2))

    def __getitem__(self, idx):
        batch = {"sentence1": self.sentences1[idx],
                 "sentence2": self.sentences2[idx],
                 }

        return batch

    # @cache
    def make_data(self, filepathList):
        sentences1 = []
        sentences2 = []
        for filepath in filepathList:
            inputs_text = []
            fileList = os.listdir(filepath)
            fileList.sort(key=lambda keys: [pinyin(i, style=Style.TONE3) for i in keys])
            # j = 0
            for file in fileList:
                # if j > 1:
                #     print(file)
                #     break
                # sentence = ""
                # isNext = False
                f = open(os.path.join(filepath, file), encoding="UTF-8")
                for line in f:
                    # if isNext:
                    #     inputs_text.append(line + self.config.end_token)
                    # if len(line) < 150 and len(sentence) < 200:
                    #     sentence += line
                    #     # isNext = False
                    #     # print(sentence)
                    #     continue
                    # if line == "……\n":
                    #     continue
                    inputs_text.append((line + self.config.end_token).replace("\n", "")
                                       .replace("\u3000", "").replace("\t", ""))
                    # isNext = True
                    sentence = ""
                # j += 1
            for i in tqdm(range(int(len(inputs_text) - 1)), desc=filepath):
                # # 如果当合并的句子加上当前的句子加上下一个句子没到上限的话
                # # 就把当前句子合并进去然后继续循环
                # if len(sentence1 + inputs_text[i] + inputs_text[i + 1]) < 125:
                #     sentence1 += inputs_text[i]
                #     if i == len(inputs_text) - 2:
                #         sentence2 = inputs_text[i + 1]
                #     else:
                #         continue
                # # 合并的句子加上当前的句子超过上限只可能是
                # # 当前句子超过上限并且为第一个要合并的句子
                # # 跳过
                # if len(inputs_text[i]) > 125:
                #     continue
                # # 如果当前句子加下一句超过上限
                # # 就拆分当前句子来作预测
                # if len(inputs_text[i] + inputs_text[i + 1]) > 125 and sentence1 == "":
                #     sentence1 = inputs_text[:int(len(inputs_text) / 2)]
                #     sentence2 = inputs_text[int(len(inputs_text) / 2):]
                # if sentence2 == "":
                #     sentence2 = inputs_text[i]
                # sentences1.append(str(sentence1))
                # sentences2.append(str(sentence2))
                # sentence1 = ""
                # sentence2 = ""
                # if len(inputs_text[i]) > 5 and len(inputs_text[i + 1]) > 5:
                sentences1.append(inputs_text[i])
                sentences2.append(inputs_text[i + 1])
        all_data = {'sentences1': sentences1, 'sentences2': sentences2}
        return all_data

    def generate_batch(self, data_batch):
        sentences1, sentences2 = [], []
        for data in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            sentences1.append(data["sentence1"])
            sentences2.append(data["sentence2"])

        token = self.tokenizer(sentences1, sentences2, padding=True, truncation=True, max_length=512,
                               add_special_tokens=False,
                               return_tensors="pt")
        inputs = token["input_ids"]
        token_type_ids = token["token_type_ids"]
        inputs, labels = self.mask_tokens(inputs, token_type_ids.bool())
        # inputs, labels = self.mask_tokens(inputs)
        attention_mask = compute_attention_mask(token_type_ids)
        return inputs, attention_mask, labels

    def pad_sequence(self, sequences, max_len=None, padding_value=0):
        """
        对一个List中的元素进行padding
        """
        if max_len is None:
            max_len = max([s.size(0) for s in sequences])
        out_tensors = []
        for tensor in sequences:
            if tensor.size(0) < max_len:
                tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
            else:
                tensor = tensor[:max_len]
            out_tensors.append(tensor)
        out_tensors = torch.stack(out_tensors, dim=1)
        return out_tensors

    def mask_tokens(self, inputs, segment_ids=None):
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.config.mlm_probability)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        if segment_ids != None:
            probability_matrix.masked_fill_(~segment_ids, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100

        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels


def compute_attention_mask(segment_ids):
    idxs = torch.cumsum(segment_ids, dim=1)
    mask = idxs[:, None, :] <= idxs[:, :, None]
    mask = mask.to(torch.float32)
    return mask


def train(model, train_dataloader, config):
    """
    训练
    :param model: nn.Module
    :param train_dataloader: DataLoader
    :param config: Config
    """

    model.to(config.device)

    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")

    if os.path.exists(os.path.join(config.save_path, 'fiction_model.bin')):
        model.load_state_dict(
            torch.load(os.path.join(config.save_path, 'fiction_model.bin')))
        print("## 成功载入已有模型进行训练......")
    else:
        print("## 未找到已有模型，开始训练......")
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)

    amp.register_float_function(torch, 'sigmoid')
    amp.register_float_function(torch, 'softmax')
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    print("Start training...Train {} epoch".format(config.epochs))
    model.train()
    for cur_epc in range(int(config.epochs)):
        training_loss = 0
        for step, (input_ids, attention_mask, labels) in enumerate(
                tqdm(train_dataloader, desc='Epoch: {}'.format(cur_epc + 1))):
            input_ids = input_ids.to(config.device)
            attention_mask = attention_mask.to(config.device)
            labels = labels.to(config.device)
            loss = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask).loss

            optimizer.zero_grad()
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            # model.zero_grad()
            training_loss += loss.item()
        print("Training loss: ", training_loss)

    print("Training finished.")
    if not os.path.exists(f'{config.save_path}'):
        os.makedirs(f'{config.save_path}')
    torch.save(model.state_dict(), os.path.join(config.save_path, 'fiction_model.bin'))
    print("Model saved.")


def eval(model, tokenizer, config):
    sen = "“唉…”想起下午的测试，萧炎轻叹了一口气，懒懒的抽回手掌，双手枕着脑袋，眼神有些恍惚…"
    if os.path.exists(os.path.join(config.save_path, 'fiction_model.bin')):
        state = torch.load(os.path.join(config.save_path, 'fiction_model.bin'))
        model.load_state_dict(state)
        print("## 成功载入已有模型进行推理......")
    else:
        raise ValueError(f"模型 {config.model_save_path} 不存在！")
    model.to(config.device)
    num = 300
    model.eval()
    sen1 = ""
    for i in range(num):
        token = tokenizer(sen + '[EOS]', sen1 + "[MASK]", add_special_tokens=False, return_tensors='pt')
        input = token['input_ids'].to(config.device)
        token_type_ids = token["token_type_ids"]
        attention_mask = compute_attention_mask(token_type_ids)
        attention_mask = attention_mask.to(config.device)
        output = model(input_ids=input, attention_mask=attention_mask).logits
        # print(output.size())
        pred = torch.argmax(output, dim=-1)
        # print(pred.size())
        pred = pred.data.cpu().numpy().tolist()[0][-1]
        pred = tokenizer.decode(pred)
        sen1 = sen1 + pred
        if pred == config.end_token:
            print("预测结束")
            break
    print(sen)
    print(sen1)


if __name__ == '__main__':
    config = Config(batch_size=16, epochs=5, learning_rate=1e-6, weight_decay=0,
                    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
                    from_path='../bert-base-chinese',
                    save_path='../fiction-bert-base-chinese', mlm_probability=0.80)
    bert_tokenizer = BertTokenizerFast.from_pretrained(config.from_path)

    bert_fiction_model = BertForMaskedLM.from_pretrained(config.from_path)
    filePathList = ["../斗破苍穹", "../大奉打更人", "../赘婿", "../诡秘之主", "../我有一座冒险屋", "../花千骨",
                    "../全职法师", "../修真聊天群"
        , "../雪中悍刀行", "../黄金瞳", "../全职高手", "../斗罗大陆"]
    # filePathList = ["../斗破苍穹"]
    train_dataset = TrainDataset(filePathList, bert_tokenizer, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=train_dataset.generate_batch)
    # train(model=bert_fiction_model, train_dataloader=train_dataloader, config=config)
    eval(model=bert_fiction_model, tokenizer=bert_tokenizer, config=config)
    # “唉…”想起下午的测试，萧炎轻叹了一口气，懒懒的抽回手掌，双手枕着脑袋，眼神有些恍惚…
    # 在萧炎的思绪之中，一道道身影缓缓的从那道身影身后走出，然后在那道身影身后，一道身影，正是一名身着黑袍的男子，他身材高大，脸庞上，有着一抹笑容，这男子，正是萧炎。[EOS]
    # sen = ['[CLS]……想起下午的测试，萧炎轻叹了一口气，懒懒的抽回手掌，双手枕着脑袋，眼神有些恍惚...[EOS]',
    #        "我嗡嗡嗡嗡嗡嗡嗡嗡嗡嗡[EOS][EOS][EOS][EOS][EOS]"]
    # input = bert_tokenizer(sen[0], sen[1], add_special_tokens=False, return_tensors='pt', padding=True, truncation=True)
    # print(input['input_ids'])
    # print(train_dataset.mask_tokens(input['input_ids'], input['token_type_ids'].bool()))
