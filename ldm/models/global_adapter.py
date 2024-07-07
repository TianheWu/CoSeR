import torch
from torch import nn
from einops import rearrange

from ldm.models.Qformer import BertConfig, BertLMHeadModel, BertModel

class CogAdapter(nn.Module):
    def __init__(self, num_query_token, vision_width, cross_attention_freq=2, num_hidden_layers=12):
        super().__init__()
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.hidden_size = vision_width
        encoder_config.num_attention_heads = 16
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.num_hidden_layers = num_hidden_layers
        self.Qformer = BertModel(encoder_config, add_pooling_layer=False)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        self.num_query_token = num_query_token
    
    def forward(self, x, text=None, tokens=None):
        image_atts = torch.ones(x.size()[:-1], dtype=torch.long).to(
            x.device
        )

        query_output = self.Qformer(
            query_embeds=self.query_tokens,
            encoder_hidden_states=x,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        output = query_output.last_hidden_state

        if text is not None:
            tokens_num  = self.num_query_token
            temp_tokens = tokens
            temp_semantic = text
            cls_num = temp_tokens.argmax(dim=-1) + 1
            temp_results = []
            for i in range(temp_semantic.shape[0]):
                if cls_num[i] >= tokens_num:
                    temp_result = temp_semantic[i, (cls_num[i]-tokens_num):(cls_num[i])]
                else:
                    temp_result = torch.cat([temp_semantic[i, :(cls_num[i])], temp_semantic[i, cls_num[i]-1].unsqueeze(0).expand(tokens_num-cls_num[i], -1)], 0)
                temp_results.append(temp_result)
            text = torch.stack(temp_results, 0)
            loss = nn.functional.mse_loss(output, text)
            return output, loss
        else:
            return output
