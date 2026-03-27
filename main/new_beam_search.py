import my_model
from tokenizer import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
torch.backends.cudnn.enabled = False
import torch.utils.data.distributed
torch.autograd.set_detect_anomaly(True)


def build_model_and_optimizer(lr=0.0001):
    vocab_length = 80
    d_model = 512
    heads = 8
    blocks = 6
    max_length = 300
    model = my_model.Prior_Transformer(vocab_length,d_model,heads,blocks,max_length,device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.998))
    return model, optimizer
class Inference_model(nn.Module):
    def __init__(self,Model):
        super().__init__()
        self.model=Model
    def _ENCODER(self,x,matrix):
        x = self.model.embedding(x) + self.model.position[:, :x.shape[1]]
        feature = self.model.new_encoder(x, matrix[0],matrix[1])
        return feature
    def _DECODER(self,x1,feature):
        x1 = self.model.embedding(x1) + self.model.position1[:, :x1.shape[1]]
        out = self.model.Decoder(x1, feature)
        return self.model.fc(out)

def inference(src_tensor,matrix_tensor,  model, beam_size=5,device=None):
    batch_size = len(src_tensor)
    # 1.get feature_batch
    # src_tensor=F.one_hot(src_tensor.long(),num_classes=411).float()
    feature = model._ENCODER(src_tensor,matrix_tensor)
    #  extention feature
    shape = feature.shape
    expanded_feature = feature.repeat(1, beam_size, 1).reshape(shape[0] * beam_size, shape[1], shape[2])

    min_length = 0
    max_sentence_length = 300
    end_token = 2
    n_best = beam_size
    top_beam_finished = torch.zeros([batch_size], dtype=torch.uint8)
    batch_offset = torch.arange(batch_size, dtype=torch.long)

    beam_offset = torch.arange(
        0,
        batch_size * beam_size,
        step=beam_size,
        dtype=torch.long,
        device=device)
    alive_seq = torch.full(
        [batch_size * beam_size, 1],
        1,
        dtype=torch.long,
        device=device)
    # Give full probability to the first beam on the first step.
    topk_log_probs = (
        torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                     device=device).repeat(batch_size))
    if True:
        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        memory_bank = expanded_feature
        for step in range(max_sentence_length):
            # decoder_input = alive_seq[:, -1].view(1, -1, 1)
            # Decoder forward.
            # alive_seq=alive_seq.long()
            dec_out = model._DECODER( alive_seq,memory_bank)  # shape=[b*k,s,e]
            # Generator forward.
            log_probs = torch.log10(F.softmax(dec_out[:, step, :], dim=1))  # shape=[b*k,e]

            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, end_token] = -1e20
            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)
            # Flatten probs into a list of possibilities.
            curr_scores = log_probs
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)
            # Recover log probs.
            topk_log_probs = topk_scores
            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)
            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)
            temp_device = select_indices.device
            select_indices = torch.tensor(select_indices.tolist(), dtype=torch.int32).to(temp_device)
            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)
            is_finished = topk_ids.eq(end_token)
            if step + 1 == max_sentence_length:
                is_finished.fill_(1)
            # Save finished hypotheses.
            if is_finished.any():
                # Penalize beams that finished.
                # topk_log_probs.masked_fill_(is_finished, -1e10)
                is_finished = is_finished.to('cpu')
                top_beam_finished |= is_finished[:, 0].eq(1)
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                non_finished_batch = []
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        # if (predictions[i, j, 1:] == end_token).sum() <= 1:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]  # Ignore start_token.
                        ))
                    # End condition is the top beam finished and we can return
                    # n_best hypotheses.
                    if top_beam_finished[i] and len(hypotheses[b]) >= n_best:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                    else:
                        non_finished_batch.append(i)
                non_finished = torch.tensor(non_finished_batch)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                top_beam_finished = top_beam_finished.index_select(
                    0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                non_finished = non_finished.to(topk_ids.device)
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                select_indices = batch_index.view(-1)
                ##修改内容
                temp_device = select_indices.device
                select_indices = torch.tensor(select_indices.tolist(), dtype=torch.int32).to(temp_device)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            memory_bank = memory_bank.permute(1, 0, 2)
            memory_bank = memory_bank.index_select(1, select_indices)
            memory_bank = memory_bank.permute(1, 0, 2)
        return results['predictions']

def save(tgt, save_path, tokenizer):
    for batch in tgt:
        if len(batch) == 0:  # 防止出现空集的情况
            print('Empty')
            batch = [torch.tensor([9, 2]) for i in range(10)]
        for Beam in batch:
            beam = Beam.cpu().tolist()
            beam = tokenizer.detokenize(beam)
            try:
                end = beam.index('[EOS]')
            except:
                end = 201
            sentence = beam[0:end]
            if sentence == []:
                sentence = ['C']
            sentence = ' '.join(sentence)
            with open(save_path, 'a+', encoding='utf8') as f:
                f.write(sentence + '\n')

def beamSearch(model, data_loader, beam_size, save_path,tokenizer,device):
    Infer_model=Inference_model(model).to(device)
    with torch.no_grad():
        i = 0
        for src,_,matrix_energy,matrix_bondlength in data_loader:
            i+=1
            src=src.to(device)
            matrix_energy=matrix_energy.to(device)
            matrix_bondlength = matrix_bondlength.to(device)
            result = inference(src_tensor=src,matrix_tensor=(matrix_energy,matrix_bondlength),  model=Infer_model, beam_size=beam_size,device=device)
            save(result,save_path,tokenizer)

class logger():
    def __init__(self, file):
        self.file = file
    def get_time(self):
        t = time.time()
        time_struct = time.localtime(t)
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time_struct)
        return formatted_time
    def info(self, text):
        t = self.get_time()
        text = f'[Time {t}] ' + text
        print(text)
        with open(self.file, 'a+', encoding='utf-8') as f:
            f.write(text + '\n')
if __name__=='__main__':
    vocab_file = '../vocab/vocab_with_error.pt'
    tokenizer = SimpleTokenizer(vocab_file)
    batch_size = 32
    test_set = NLPDataset('../data/tgt-test.txt', '../data/src-test.txt', tokenizer, max_length=300)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import calculate_topk_reverse
    save_path = '../experiment/prediction_S165-30-1.txt'
    beam_size=10
    model = torch.load('../experiment/S165-30-1_model_60000.pt',weights_only=False)
    beamSearch(model, test_loader, beam_size, save_path,tokenizer,device)
    print('done!')
    result = []
    data_path = '../data/src-test.txt'
    for k in range(10):
        result.append(
            f'top-{k + 1}={calculate_topk_reverse.calculateTopk(k=k + 1, data_path=data_path, save_path=save_path)}')
    for i in result:
        logger.info(i)














