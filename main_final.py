import numpy as np
import pandas as pd
from data_loader import file2data, TimDataset
import torch
from torch.utils.data import DataLoader

# Network
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from sklearn.metrics import confusion_matrix, average_precision_score  # classification_report, matthews_corrcoef
import copy
import shap


def calc_hter(conf_mat):
    tn = conf_mat[0, 0]
    fp = conf_mat[0, 1]
    fn = conf_mat[1, 0]
    tp = conf_mat[1, 1]
    mcc = (tp * tn - fn * fp) / np.sqrt((tn + fn) * (tn + fp) * (tp + fn) * (tp + fp))
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    hter = (far + frr) / 2 * 100

    # recall = tp/(tp + fn)
    # precision = tp/(tp + tn)
    # auc_precision_recall = auc(recall, precision)

    return round(hter, 2), round(mcc, 3)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)


def model_freeze(models):
    model_ft = models # .resnet50(pretrained=True)
    ct = 0
    for child in model_ft.children():
        ct += 1
        if ct < 7:
            for param in child.parameters():
                param.requires_grad = False
    return model_ft


class FilePath:
    root_dir = './CLL_TIM_Sandbox/'
    yaml_fn = 'file_names.yaml'
    save_path = './results/'
    fn_mcc_valid = save_path + 'final_mcc_valid'
    fn_mcc_test = save_path + 'final_mcc_test'
    res_all = save_path + 'res_valid'
    res_all_tr = save_path + 'res_train'


use_cuda = torch.cuda.is_available()
# use_cuda = False


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if use_cuda:
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if use_cuda:
        return x.cpu().data.numpy()
    return x.data.numpy()


class AutoEncoder(nn.Module):
    def __init__(self, hidden_units, num_features, latent_features=2):
        super(AutoEncoder, self).__init__()
        # We typically employ an "hourglass" structure
        # meaning that the decoder should be an encoder
        # in reverse.

        self.encoder = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=hidden_units),
            nn.ReLU(),
            # bottleneck layer
            nn.Linear(in_features=hidden_units, out_features=latent_features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_features, out_features=hidden_units),
            nn.ReLU(),
            # output layer, projecting back to image size
            nn.Linear(in_features=hidden_units, out_features=num_features)
        )

    def forward(self, x):
        outputs = {}
        # we don't apply an activation to the bottleneck layer
        z = self.encoder(x)

        # apply sigmoid to output to get pixel intensities between 0 and 1
        # x_hat = torch.sigmoid(self.decoder(z))
        x_hat = self.decoder(z)

        return {
            'z': z,
            'x_hat': x_hat
        }


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, cat_cols, cont_cols):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols

    def __call__(self, sample):
        features, outcomes = sample['features'], sample['outcomes']
        features = features.astype(np.float32)
        outcomes = outcomes.astype(np.float32)
        # return {'features': torch.from_numpy(features.values),
        #        'outcomes': torch.from_numpy(outcomes.values)}
        cont_features = torch.from_numpy(features[self.cont_cols].values)
        if self.cat_cols:
            cat_features = torch.from_numpy(features[self.cat_cols].values)
        else:
            cat_features = torch.from_numpy(features[self.cat_cols].values)
        return cont_features, cat_features, torch.from_numpy(outcomes.values)


class CLLNet(nn.Module):
    def __init__(self, num_classes, num_features, num_outcomes, encoder):
        super(CLLNet, self).__init__()
        self.encoder = encoder

        self.num_classes = num_classes

        # Treatment [128 64], l1_lambda = 1e-2, l2_lambda = 1e-4
        # infection [64 32], l1_lambda = 1e-3, l2_lambda = 1e-5
        # Death [128 64], l1_lambda = 1e-3, l2_lambda = 1e-4

        nh1 = 128  # S* num_outcomes  # 256  # 64
        nh2 = 128  # 32
        # nh3 = 64
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2_drop = nn.Dropout(p=0.5)
        # self.fc3_drop = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(num_features=nh1, affine=True)
        self.bn2 = nn.BatchNorm1d(num_features=nh2, affine=True)
        # self.bn3 = nn.BatchNorm1d(num_features=nh3)
        # fully connected output layers
        self.fc1_features = num_features  # num_features
        self.fc1 = nn.Linear(in_features=self.fc1_features, out_features=nh1)

        # self.fc3 = nn.Linear(in_features=nh2, out_features=nh3)
        fc2_temp = []
        fc_final_temp = []
        for i in range(num_outcomes):
            fc2_temp.append(nn.Linear(in_features=nh1, out_features=nh2))
            fc_final_temp.append(nn.Linear(in_features=nh2, out_features=num_classes))

        self.fc2 = nn.ModuleList(fc2_temp)
        self.fc_final = nn.ModuleList(fc_final_temp)

    def forward(self, x_cont, x_cat):
        # Your code here!
        # x = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        # x = x.view(-1, self.fc1_features)  # x = x.view(x.size(0), -1)
        if self.encoder:
            x_cat = self.encoder(x_cat)
            x = torch.cat((x_cont, x_cat), dim=1)
        else:
            x = x_cont
        # print(x.device)
        x = F.relu(self.fc1_drop(self.bn1(self.fc1(x))))
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.fc1(x))

        # x = F.relu(self.fc3_drop(self.bn3(self.fc3(x))))
        # note use of Functional.dropout, where training must be explicitly defined (default: False)
        # x = F.dropout(x, p=0.5, training=self.training)
        num_outcomes = len(self.fc_final)
        # z = ['']*num_outcomes
        for i in range(num_outcomes):
            z = F.relu(self.fc2_drop(self.bn2(self.fc2[i](x))))
            # z = F.relu(self.bn2(self.fc2[i](x)))
            z = self.fc_final[i](z)
            if i == 0:
                zf = z
            else:
                zf = torch.cat((zf, z), dim=0)
        zf = torch.reshape(zf, tuple([num_outcomes] + list(z.size())))
        # x = F.log_softmax(x, dim=1)
        # x = F.sigmoid(x)
        return zf


def validate(cll_net, loader):
    cll_net.eval()
    # evaluation
    num_outcomes = len(cll_net.fc_final)
    correct = [0] * num_outcomes
    total = [0] * num_outcomes
    mcc = [0] * num_outcomes
    hter = [0] * num_outcomes
    pr_auc = [0] * num_outcomes
    y_score = [[] for i in range(num_outcomes)]
    y_pred = [[] for i in range(num_outcomes)]
    y_true = [[] for i in range(num_outcomes)]
    for data in loader:
        inputs_cont, inputs_cat, labels = data
        outputs = cll_net(get_variable(inputs_cont), get_variable(inputs_cat))  # fix this
        # _, predicted = torch.max(outputs.data, 1)
        for io in range(num_outcomes):
            # alpha = 0.5 * (pos_weight[io] - 1)/(pos_weight[io] + 1)
            out = outputs[io, :, :].data
            y_score[io] += out.cpu().numpy().transpose().tolist()[0]
            predicted = torch.round(torch.sigmoid(outputs[io, :, :].data.cpu()))
            total[io] += labels[:, io].size(0)
            # correct += (predicted == torch.transpose(labels, 0, 1)).sum()
            correct[io] += (predicted.squeeze() == labels[:, io]).sum()
            y_pred[io] += predicted.numpy().transpose().tolist()[0]  # predicted.numpy().transpose().tolist()[0]
            y_true[io] += labels[:, io].numpy().tolist()  # labels[:, io].numpy().transpose().tolist()
    for io in range(num_outcomes):
        conf_mat = confusion_matrix(y_true[io], y_pred[io])
        hter[io], mcc[io] = calc_hter(conf_mat)
        pr_auc[io] = round(average_precision_score(y_true[io], y_score[io]), 3)
        print('Accuracy of the network on the {} test samples: {:4.2f} %, MCC = {:4.3f}, HTER = {:4.2f}, PRAUC = {:4.3f}'.format(
            total[io], 100 * correct[io].true_divide(total[io]), mcc[io], hter[io], pr_auc[io]))

    return mcc, hter, pr_auc


def main():

    print("Running GPU.") if use_cuda else print("No GPU available.")

    n_lvl_th = 3
    ae_hn = 64
    ae_lt = 64
    do_pre = True  # impute missing values and scale (mean and std)
    sep_cols = False
    outcome_str = ['Treatment', 'Infection','Death']
    num_outcomes = len(outcome_str)
    features, outcomes, cat_cols, cont_cols = file2data(root_dir=FilePath.root_dir, yaml_fn=FilePath.yaml_fn, do_pre=do_pre,
                                                        n_lvl_th=n_lvl_th, outcome_str=outcome_str, sep_cols=sep_cols)

    print('number of continuous and categorical features: {:4n}, {:4n}'.format(len(cont_cols), len(cat_cols)))

    # initialize random seed
    # torch.backends.cudnn.deterministic = True
    seed = 33
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    seed = 330
    train_set = TimDataset(features['x_train'], outcomes['y_train'], transform=ToTensor(cat_cols, cont_cols))
    valid_set = TimDataset(features['x_valid'], outcomes['y_valid'], transform=ToTensor(cat_cols, cont_cols))
    test_set = TimDataset(features['x_test'], outcomes['y_test'], transform=ToTensor(cat_cols, cont_cols))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2)

    # number of features
    if sep_cols:
        num_features = len(cont_cols) + ae_lt  # len(features['x_train'].columns)  # get number of features
    else:
        num_features = len(cont_cols)

    outcomes_list = [o + ' Outcome (2yr)' for o in outcome_str]
    pos_weight = [0]*num_outcomes
    for io in range(num_outcomes):
        pos_weight[io] = (outcomes['y_train'][outcomes_list[io]] == 0).sum() / (outcomes['y_train'][outcomes_list[io]] == 1).sum()

    print(pos_weight)

    del features, outcomes  # delete features and outcomes

    # Define the auto-encoder
    ae_net = AutoEncoder(hidden_units=ae_hn, num_features=len(cat_cols), latent_features=ae_lt)

    # xnet = model_freeze(ae_net)
    # Define network
    # Setup encoder
    if sep_cols:

        encoder = copy.deepcopy(ae_net.encoder)
        for params in encoder.parameters():
            # print('params:',  params)
            params.requires_grad = False  # Freeze all weights
    else:
        encoder = []

    cll_net = CLLNet(num_classes=1, num_features=num_features, num_outcomes=num_outcomes, encoder=encoder)
    if use_cuda:
        cll_net.cuda()
        criterion = [nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight[io]]).cuda()) for io in range(num_outcomes)]
    else:
        criterion = [nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight[io]])) for io in range(num_outcomes)]
    # cll_net.apply(init_weights)
    # Define loss and optimizer
    # criterion = [nn.BCEWithLogitsLoss(pos_weight=pos_weight[io]) for io in range(num_outcomes)]
    # criterion = [nn.BCEWithLogitsLoss() for io in range(num_outcomes)]

    # criterion[io] = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([pos_weight[0]]))
    # BCELoss()  # NLLLoss()  # CrossEntropyLoss()  # Your code here!
    optimizer = optim.Adam(cll_net.parameters(), lr=5e-5)  # Your code here!
    # optimizer = optim.SGD(cll_net.parameters(), lr=1e-2) #, weight_decay=0.01)

    # Train
    l1_lambda = 2 # 50  # 1e-2  # 1e-2  # 1e-2
    l2_lambda = 0.01 # 5  # 1e-5  # 1e-5
    best_mcc = [-1] * num_outcomes
    no_imp_since = [0] * num_outcomes  # number of epochs no improvement of character error rate occurred
    early_stopping = 5  # stop training after this number of epochs without improvement
    num_epoch = 40
    res_all = []
    res_all_tr = []
    for epoch in range(num_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        cll_net.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs_cont, inputs_cat, labels = data

            # wrap them in Variable
            # ?? Commented! inputs, labels = Variable(inputs), Variable(labels)

            # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) # cuda
            # zero the parameter gradients
            # Your code here!
            optimizer.zero_grad()

            # forward + backward + optimize
            # Your code here!
            # print(next(cll_net.parameters()).device)
            output = cll_net(get_variable(inputs_cont.float()), get_variable(inputs_cat.float()))
            # criterion.weight = torch.FloatTensor([class_weight[k] for k in torch.transpose(labels, 0, 1).tolist()[0]])
            t_loss = ['']*num_outcomes

            for io in range(num_outcomes):
                # criterion[io].pos_weight = torch.FloatTensor([pos_weight[io]])
                t_loss[io] = criterion[io](output[io, :, :].squeeze(), get_variable(labels[:, io]))
            num_of_parameters = sum(map(torch.numel, cll_net.parameters()))
            # print(num_of_parameters)
            l1_norm = sum(p.abs().sum() for p in cll_net.parameters())/num_of_parameters  # / sum(p.size(dim=0) for p in cll_net.parameters())
            l2_norm = sum(p.pow(2).sum() for p in cll_net.parameters())/num_of_parameters  # / sum(p.size(dim=0) for p in cll_net.parameters())
            loss = sum(t_loss)/num_outcomes + l1_lambda * l1_norm + l2_lambda * l2_norm
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss
            if i % 5 == 4:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

        mcc, hter, pr_auc = validate(cll_net, valid_loader)
        mcc_tr, hter_tr, pr_auc_tr = validate(cll_net, train_loader)
        res_all.append(mcc + hter + pr_auc)
        res_all_tr.append(mcc_tr + hter_tr + pr_auc_tr)
        # if best validation accuracy so far, save model parameters
        # mcc_avg = np.mean(mcc)
        es_count = 0
        for j in range(num_outcomes):
            if (mcc[j] > best_mcc[j]) and (no_imp_since[j] < early_stopping):
                print(outcome_str[j] + ': MCC improved, save model')
                best_mcc[j] = mcc[j]
                no_imp_since[j] = 0
                torch.save(cll_net.state_dict(), FilePath.save_path + '_'.join(outcome_str) + '_outcome_' + outcome_str[j] + '_' + str(seed) + '.pth')
                source_file = open(FilePath.save_path + '_'.join(outcome_str) + '_outcome_' + outcome_str[j] + '_' + str(seed) + '.summary', 'w')
                print(cll_net, file=source_file)
                source_file.close()
                str_out = ''
                for io in range(num_outcomes):
                    str_out += outcome_str[io] + ', MCC, HTER, and PRAUC of saved model on validation set: %.3f, %.2f%%, %.3f\n' % (mcc[io], hter[io], pr_auc[io])
                open(FilePath.fn_mcc_valid + '_'.join(outcome_str) + '_outcome_' + outcome_str[j] + '_' + str(seed) + '.txt', 'w').write(str_out)
            else:
                print(outcome_str[j] + ': MCC not improved')
                no_imp_since[j] += 1

            # stop training if no more improvement in the last x epochs
            if no_imp_since[j] >= early_stopping:
                print(outcome_str[j] + ': No more improvement since %d epochs. Training stopped.' % early_stopping)
                es_count += 1
        if es_count == num_outcomes:
            break

    columns = [s + '_' + m for s in ['mcc', 'hter', 'pr_auc'] for m in outcome_str]
    df = pd.DataFrame(res_all, columns=columns)
    df.to_csv(FilePath.res_all + '_'.join(outcome_str) + '_outcome_' + outcome_str[j] + '_' + str(seed) + '.csv', index=False)
    df = pd.DataFrame(res_all_tr, columns=columns)
    df.to_csv(FilePath.res_all_tr + '_'.join(outcome_str) + '_outcome_' + outcome_str[j] + '_' + str(seed) + '.csv', index=False)

    print('Finished Training')

    # Test
    # load model(s)
    for j in range(num_outcomes):
        cll_net = CLLNet(num_classes=1, num_features=num_features, num_outcomes=num_outcomes, encoder=encoder)
        cll_net.load_state_dict(torch.load(FilePath.save_path + '_'.join(outcome_str) + '_outcome_' + outcome_str[j] + '_' + str(seed) + '.pth'))
        cll_net.cuda()
        mcc, hter, pr_auc = validate(cll_net, test_loader)
        str_out = ''
        for io in range(num_outcomes):
            str_out += outcome_str[io] + ', MCC, HTER, and PRAUC of saved model on validation set: %.3f, %.2f%%, %.3f\n' % (
            mcc[io], hter[io], pr_auc[io])
        open(FilePath.fn_mcc_test + '_'.join(outcome_str) + '_outcome_' + outcome_str[j] + '_' + str(seed) + '.txt', 'w').write(str_out)



if __name__ == '__main__':
    main()
