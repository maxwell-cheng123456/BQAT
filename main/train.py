import my_model
from tokenizer import *
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import math
torch.backends.cudnn.enabled = False
torch.manual_seed(40)
import torch.utils.data.distributed
torch.autograd.set_detect_anomaly(True)

#  模型配置
#  hyperparameters configuration
class Config():
    def __init__(self):
        self.lr=0.0001
        self.using_gpu = True
        self.experiment = 'BQAT'
        self.training_steps = 60000
        self.warm_up=6000
        self.batch_size = 128
        self.data_file = 'data'
        self.vocab_length = 80
        self.d_model = 512
        self.heads = 8
        self.blocks = 6
        self.max_length = 300
        self.vocab_file = 'vocab_with_error'
        self.save_temp_models = False
        self.beam_size=10

config = Config()


# 加载数据集
# load dataset
print('Start loading dataset...')
vocab_file = f'../vocab/{config.vocab_file}.pt'
tokenizer = SimpleTokenizer(vocab_file)
train_set = NLPDataset(f'../{config.data_file}/tgt-train.txt', f'../{config.data_file}/src-train.txt', tokenizer,
                       max_length=config.max_length)
batch_size = config.batch_size
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_set = NLPDataset(f'../{config.data_file}/tgt-val.txt', f'../{config.data_file}/src-val.txt', tokenizer, max_length=config.max_length)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_set = NLPDataset(f'../{config.data_file}/tgt-test.txt', f'../{config.data_file}/src-test.txt', tokenizer,
                      max_length=config.max_length)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
if config.using_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device('cpu')

print('Load dataset successfully')




def build_model_and_optimizer():
    logger.info('Starting building model...')
    vocab_length = config.vocab_length
    d_model = config.d_model
    heads = config.heads
    blocks = config.blocks
    max_length = config.max_length
    model = my_model.Prior_Transformer(vocab_length,d_model,heads,blocks,max_length,device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.998))
    logger.info('Build model successfully!')
    return model, optimizer

# 日志文件操作
# logger file and operation
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

# 测试函数
# Test function
def multi_test(model, test_loader):
    model.eval()
    PPL = 0
    total = 0
    correct_top_1 = 0
    correct_top_5 = 0
    i = 0
    for src, tgt,matrix_energy,matrix_bondlength in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        matrix_energy = matrix_energy.to(device)
        matrix_bondlength = matrix_bondlength.to(device)
        i += 1
        true_labels = tgt[:, 1:]
        prob = model(src, tgt[:, :-1],matrix_energy,matrix_bondlength)
        prob = prob.view(prob.shape[0] * prob.shape[1], prob.shape[2])
        tgt = tgt[:, 1:].reshape(-1)
        PPL += torch.exp(F.cross_entropy(prob, tgt)).detach().item()

        probabilities = torch.softmax(prob, dim=-1)
        _, predicted_top_1 = torch.max(probabilities, dim=-1)
        _, predicted_top_5 = torch.topk(probabilities, k=5, dim=-1)

        total += true_labels.size(0) * true_labels.size(1)
        predicted_top_1 = predicted_top_1.reshape(true_labels.size(0), true_labels.size(1))
        predicted_top_5 = predicted_top_5.reshape(true_labels.size(0), true_labels.size(1), predicted_top_5.size(1))
        correct_top_1 += (predicted_top_1 == true_labels).sum().item()
        correct_top_5 += (predicted_top_5 == true_labels.unsqueeze(2)).any(dim=-1).sum().item()

    PPL_score = PPL / i
    top_1_accuracy = correct_top_1 / total
    top_5_accuracy = correct_top_5 / total
    return PPL_score, top_1_accuracy, top_5_accuracy


# 添加Warm Up策略
# Add warm up strategy
def adjust_learning_rate(optimizer, epoch, total_epochs, lr, warm_up):
    warmup_epochs = warm_up
    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
    else:
        lr_scale = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * lr_scale

#  训练过程主函数
#  Training function

def train(model, optimizer, logger, experiment):
    logger.info('Starting training...')
    training_steps = config.training_steps
    step = 0
    while step < training_steps:
        model.train()
        LOSS = 0
        i = 0
        for src, tgt,matrix_energy,matrix_bondlength in train_loader:
            adjust_learning_rate(optimizer, step, training_steps, lr=config.lr, warm_up=config.warm_up)
            src = src.to(device)
            tgt = tgt.to(device)
            matrix_energy=matrix_energy.to(device)
            matrix_bondlength = matrix_bondlength.to(device)
            pre = model(src, tgt[:, :-1],matrix_energy,matrix_bondlength)
            pre = pre.reshape(pre.shape[0] * pre.shape[1], pre.shape[2])
            true_labels = tgt[:, 1:].reshape(-1)

            loss = F.cross_entropy(pre, true_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            LOSS += loss.detach().item()
            i += 1
            step += 1
            if step >= training_steps:
                break

            if (step % 10 == 0):
                logger.info(f"step={step}/{training_steps},loss={loss.detach().item()},lr={optimizer.param_groups[0]['lr']}")

            if step % 6000 == 0:
                logger.info('-------------evaluate acc on val_set------------')
                with torch.no_grad():
                    PPL_score, top_1_accuracy, top_5_accuracy = multi_test(model, valid_loader)

                logger.info(f'PPL score on val set={PPL_score}')
                logger.info(f'top_1_accuracy on val set={top_1_accuracy}')
                logger.info(f'top_5_accuracy on val set={top_5_accuracy}')
                if config.save_temp_models:
                    torch.save(model, f'../experiment/{experiment}_model_{iter}.pt')
                    logger.info(f'model_{step} saved!')
                model.train()

    logger.info('-------------evaluate acc on test_set------------')
    with torch.no_grad():
        PPL_score, top_1_accuracy, top_5_accuracy = multi_test(model, test_loader)
    logger.info(f'PPL score on test set={PPL_score}')
    logger.info(f'top_1_accuracy on test set={top_1_accuracy}')
    logger.info(f'top_5_accuracy on test set={top_5_accuracy}')
    torch.save(model, f'../experiment/{experiment}_model_{step}.pt')
    logger.info(f'model_{step} saved!')
    logger.info('training finished!')

#    评估函数
#     evaluating function
def evaluate():
    logger.info('Start evaluating...')
    import new_beam_search
    import calculate_topk_reverse
    save_path = f'../experiment/{config.experiment}_prediction.txt'
    with torch.no_grad():
        new_beam_search.beamSearch(model, test_loader, config.beam_size, save_path,tokenizer,device)

#   计算Topk数值
#   Calculate topk
    data_path = f'../{config.data_file}/src-test.txt'
    result = []
    for k in range(10):
        result.append(
            f'top-{k + 1}={calculate_topk_reverse.calculateTopk(k=k + 1, data_path=data_path, save_path=save_path)}')
    for i in result:
        logger.info(i)


if __name__ == '__main__':

    experiment = config.experiment
    print(f'{experiment} started!')
    logger = logger(f'../experiment/{experiment}_log_file.txt')
    model, optimizer = build_model_and_optimizer()
    train(model, optimizer, logger, experiment)
    logger.info('Training Finished!')
    evaluate()
    logger.info('Evaluate Finished!')

