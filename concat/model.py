import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.manual_dense = nn.Linear(config.feature_size, config.hidden_size)
        # 定义7个全连接层
        self.dense1 = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense3 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense4 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense5 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense6 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense7 = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.out_proj_new = nn.Linear(config.hidden_size + config.hidden_size, 1)
        self.out_proj_new = nn.Linear(config.hidden_size, 1)

    def forward(self, features, manual_features=None, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])  [bs,hidden_size]
        y = manual_features.float()  # [bs, feature_size]
        y = self.manual_dense(y)
        y = torch.tanh(y)

        x = torch.cat((x, y), dim=-1)
        x = F.relu(self.dense1(x))  # 第一层全连接并使用ReLU激活
        x = self.dropout(x)
        x = F.relu(self.dense2(x))  # 第二层全连接
        x = self.dropout(x)
        x = F.relu(self.dense3(x))  # 第三层全连接
        x = self.dropout(x)
        x = F.relu(self.dense4(x))  # 第四层全连接
        x = self.dropout(x)
        x = F.relu(self.dense5(x))  # 第五层全连接
        x = self.dropout(x)
        x = F.relu(self.dense6(x))  # 第六层全连接
        x = self.dropout(x)
        x = F.relu(self.dense7(x))  # 第七层全连接
        x = self.dropout(x)
        x = self.out_proj_new(x)  # 输出层

        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, inputs_ids, attn_masks, manual_features=None,
                labels=None, output_attentions=None,return_logits=False):

        if inputs_ids.dtype != torch.long:
            inputs_ids = inputs_ids.long()


        outputs = \
            self.encoder(input_ids=inputs_ids, attention_mask=attn_masks, output_attentions=output_attentions)

        last_layer_attn_weights = outputs.attentions[self.config.num_hidden_layers - 1][:, :,
                                  0].detach() if output_attentions else None


        logits = self.classifier(outputs[0], manual_features)

        if return_logits:
            return logits  # 返回 logits 原生值
        else:
            prob = torch.sigmoid(logits)
            if labels is not None:
                loss_fct = BCELoss()
                loss = loss_fct(prob, labels.unsqueeze(1).float())
                return loss, prob, last_layer_attn_weights
            else:
                return prob



    def get_loss_weight(self,labels, weight_dict):
        label_list = labels.cpu().numpy().squeeze().tolist()
        weight_list = []

        for lab in label_list:
            if lab == 0:
                weight_list.append(weight_dict['clean'])
            else:
                weight_list.append(weight_dict['defect'])

        weight_tensor = torch.tensor(weight_list).reshape(-1, 1).cuda()
        return weight_tensor