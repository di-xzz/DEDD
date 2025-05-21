import torch
import torch.nn as nn
import torch.nn.functional as F
import os



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)

        self.encoded_nodes = None

    def _get_new_data(self, data, selected_node_list, prob_size, B_V):

        list = selected_node_list

        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)

        new_list_len = prob_size - list.shape[1]

        index_2 = list.type(torch.long)

        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

        new_list[index_1, index_2] = -2

        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

        new_data = data

        emb_dim = data.shape[-1]

        new_data_len = new_list_len

        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)

        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])

        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))

        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

        return new_data_

    def _get_encoding(self,encoded_nodes, node_index_to_pick):

        if node_index_to_pick.dim() != 2:
            node_index_to_pick = node_index_to_pick.unsqueeze(1)
        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)

        return picked_nodes


    def forward(self, state, selected_node_list, solution, current_step,raw_data_capacity=None,):

        def probs_to_selected_nodes(probs_, split_line_, batch_size_):
            selected_node_student_ = probs_.argmax(dim=1)
            is_via_depot_student_ = selected_node_student_ >= split_line_
            not_via_depot_student_ = selected_node_student_ < split_line_

            selected_flag_student_ = torch.zeros(batch_size_, dtype=torch.int)
            selected_flag_student_[is_via_depot_student_] = 1
            selected_node_student_[is_via_depot_student_] = selected_node_student_[
                                                                is_via_depot_student_] - split_line_ + 1
            selected_flag_student_[not_via_depot_student_] = 0
            selected_node_student_[not_via_depot_student_] = selected_node_student_[not_via_depot_student_] + 1
            return selected_node_student_, selected_flag_student_


        self.capacity = raw_data_capacity.ravel()[0].item()
        batch_size = state.problems.shape[0]
        problem_size = state.problems.shape[1]
        split_line = problem_size - 1

        selected_node_list_ = selected_node_list.clone().detach() - 1
        data_ = state.problems[:,1:,:].clone().detach()
        batch_size_V = data_.shape[0]
        problem_size1 = data_.shape[1]
        new_data = data_.clone().detach()

        left_encoded_node = self._get_new_data(new_data, selected_node_list_, problem_size1, batch_size_V)
        embedded_first_node = state.problems[:,0,:].unsqueeze(1)
        if selected_node_list_.shape[1]==0:
            embedded_last_node = state.problems[:,0,:].unsqueeze(1)
        else:
            embedded_last_node = self._get_encoding(new_data, selected_node_list_[:, -1])

        out = torch.cat((embedded_first_node,left_encoded_node,embedded_last_node), dim=1)

        if self.mode == 'train':
            remaining_capacity = state.problems[:, 1, 3]

            probs1, probs2 = self.decoder(self.encoder(out,self.capacity),
                                 selected_node_list, self.capacity,remaining_capacity)

            selected_node_student1, selected_flag_student1 = probs_to_selected_nodes(probs1, split_line, batch_size)
            selected_node_student2, selected_flag_student2 = probs_to_selected_nodes(probs2, split_line, batch_size)

            selected_node_teacher = solution[:, current_step,0]
            selected_flag_teacher = solution[:, current_step, 1]
            selected_node_teacher2 = selected_node_teacher
            selected_flag_teacher2 = selected_flag_teacher

            is_via_depot = selected_flag_teacher==1
            selected_node_teacher_copy = selected_node_teacher-1
            selected_node_teacher_copy[is_via_depot]+=split_line


            prob_select_node1 = probs1[torch.arange(batch_size)[:, None], selected_node_teacher_copy[:, None]].reshape(batch_size, 1)
            loss_node1 = -prob_select_node1.type(torch.float64).log().mean()

            prob_select_node2 = probs2[torch.arange(batch_size)[:, None], selected_node_teacher_copy[:, None]].reshape(batch_size, 1)
            loss_node2 = -prob_select_node2.type(torch.float64).log().mean()

        if self.mode == 'test':

            remaining_capacity = state.problems[:, 1, 3]

            self.encoded_nodes = self.encoder(out,self.capacity)

            probs1, probs2 = self.decoder(self.encoded_nodes, selected_node_list,self.capacity, remaining_capacity)
            selected_node_student1 = probs1.argmax(dim=1)
            is_via_depot_student1 = selected_node_student1 >= split_line
            not_via_depot_student1 = selected_node_student1 < split_line
            selected_flag_student1 = torch.zeros(batch_size, dtype=torch.int)
            selected_flag_student1[is_via_depot_student1] = 1
            selected_node_student1[is_via_depot_student1] = selected_node_student1[is_via_depot_student1] - split_line + 1
            selected_flag_student1[not_via_depot_student1] = 0
            selected_node_student1[not_via_depot_student1] = selected_node_student1[not_via_depot_student1] + 1

            selected_node_teacher = selected_node_student1
            selected_flag_teacher = selected_flag_student1



            selected_node_student2 = probs2.argmax(dim=1)
            is_via_depot_student2 = selected_node_student2 >= split_line
            not_via_depot_student2 = selected_node_student2 < split_line
            selected_flag_student2 = torch.zeros(batch_size, dtype=torch.int)
            selected_flag_student2[is_via_depot_student2] = 1
            selected_node_student2[is_via_depot_student2] = selected_node_student2[is_via_depot_student2] - split_line + 1
            selected_flag_student2[not_via_depot_student2] = 0
            selected_node_student2[not_via_depot_student2] = selected_node_student2[not_via_depot_student2] + 1

            selected_node_teacher2 = selected_node_student2
            selected_flag_teacher2 = selected_flag_student2

            loss_node1 = loss_node2 = torch.tensor(0)
        return loss_node1,loss_node2,selected_node_teacher,selected_flag_teacher,selected_node_teacher2,selected_flag_teacher2,\
            selected_node_student1,selected_node_student2,selected_flag_student1,selected_flag_student2


class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num =  1
        self.embedding = nn.Linear(3, embedding_dim, bias=True)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data_,capacity):

        data = data_.clone().detach()
        data= data[:,:,:3]

        data[:,:,2] = data[:,:,2]/capacity


        embedded_input = self.embedding(data)

        out = embedded_input

        layer_count = 0
        for layer in self.layers:
            out = layer(out)
            layer_count += 1
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.feedForward = Feed_Forward_Module(**model_params)


    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 +   multi_head_out
        out2 = self.feedForward(out1)

        out3 = out1 + out2
        return out3



class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['decoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim+1, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num-1)])
        self.last_att_layer1 = DecoderLayer(**model_params)
        self.last_att_layer2 = DecoderLayer(**model_params)

        self.Linear_final1 = nn.Linear(embedding_dim, 2, bias=True)
        self.Linear_final2 = nn.Linear(embedding_dim, 2, bias=True)

    def final_process(self, out, batch_size_V, problem_size, selected_node_list_, left_encoded_node):

        props = F.softmax(out, dim=-1)
        customer_num = left_encoded_node.shape[1]
        props = torch.cat((props[:, 1:customer_num + 1], props[:, customer_num + 1 + 1 + 1:-1]),
                          dim=1)

        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)
        props = props_clone

        new_props = torch.zeros(batch_size_V, 2 * (problem_size))

        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:,None].repeat(1,selected_node_list_.shape[1]*2)

        index_2_ =torch.cat( ((selected_node_list_).type(torch.long), (problem_size)+ (selected_node_list_).type(torch.long) ),dim=-1)
        new_props[index_1_, index_2_,] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)
        new_props[index] = props.ravel()
        return new_props

    def forward(self, data,selected_node_list,capacity,remaining_capacity):

        selected_node_list_ = selected_node_list.clone().detach() - 1

        batch_size_V = data.shape[0]

        problem_size = data.shape[1] + selected_node_list.shape[1] - 2

        left_encoded_node = data[:, 1:-1, :]

        embedded_first_node = data[:, 0, :].unsqueeze(1)

        embedded_last_node = data[:,-1,:].unsqueeze(1)

        remaining_capacity = remaining_capacity.reshape(batch_size_V,1,1)/capacity
        first_node_cat = torch.cat((embedded_first_node,remaining_capacity), dim=2)
        last_node_cat = torch.cat((embedded_last_node,remaining_capacity), dim=2)

        embedded_first_node_ = self.embedding_first_node(first_node_cat)
        embedded_last_node_ = self.embedding_last_node(last_node_cat)

        embeded_all = torch.cat((embedded_first_node_,left_encoded_node,embedded_last_node_), dim=1)
        out = embeded_all

        layer_count = 0

        for layer in self.layers:

            out = layer(out)
            layer_count += 1

        out1 = self.last_att_layer1(out)
        out2 = self.last_att_layer2(out)

        out1 = self.Linear_final1(out1).squeeze(-1)
        out1[:, [0, -1], :] = out1[:, [0, -1], :] + float('-inf')
        out1 = torch.cat((out1[:, :, 0], out1[:, :, 1]), dim=1)

        out2 = self.Linear_final2(out2).squeeze(-1)
        out2[:, [0, -1], :] = out2[:, [0, -1], :] + float('-inf')
        out2 = torch.cat((out2[:, :, 0], out2[:, :, 1]), dim=1)

        props1 = self.final_process(out1,batch_size_V,problem_size,selected_node_list_,left_encoded_node)
        props2 = self.final_process(out2, batch_size_V, problem_size, selected_node_list_, left_encoded_node)
        
        return props1, props2


class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(self, input1):

        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)

        multi_head_out = self.multi_head_combine(out_concat)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 + out2
        return out3



def reshape_by_heads(qkv, head_num):

    batch_s = qkv.size(0)

    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)

    q_transposed = q_reshaped.transpose(1, 2)

    return q_transposed


def multi_head_attention(q, k, v):

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    score = torch.matmul(q, k.transpose(2, 3))

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))

    weights = nn.Softmax(dim=3)(score_scaled)

    out = torch.matmul(weights, v)

    out_transposed = out.transpose(1, 2)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)

    return out_concat


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):


        return self.W2(F.relu(self.W1(input1)))
