import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Evaluator():
    def __init__(self, sequences_indexer=None):
        self.sequences_indexer = sequences_indexer

    def get_macro_scores_inputs_tensor_targets_idx(self, tagger, inputs_tensor, targets_idx):
        outputs_idx = tagger.predict_idx_from_tensor(inputs_tensor)
        y_true = [i for sequence in targets_idx for i in sequence]
        y_pred = [i for sequence in outputs_idx for i in sequence]
        accuracy = accuracy_score(y_true, y_pred)*100
        f1 = f1_score(y_true, y_pred, average='macro')*100
        precision = precision_score(y_true, y_pred, average='macro')*100
        recall = recall_score(y_true, y_pred, average='macro')*100
        return accuracy, f1, precision, recall

    def is_tensor(self, X):
        return isinstance(X[0][0], torch.Tensor)

    def is_idx(self, X):
        return isinstance(X[0][0], int)

    def is_str(self, X):
        return isinstance(X[0][0], str)

    def get_macro_scores(self, tagger, inputs, targets):
        if self.is_tensor(inputs) and self.is_tensor(targets):
            return self.get_macro_scores_inputs_tensor_targets_tensor(tagger, inputs, targets)
        elif self.is_tensor(inputs) and self.is_idx(targets):
            return self.get_macro_scores_inputs_tensor_targets_idx(tagger, inputs, targets)
        elif self.is_idx(inputs) and self.is_idx(targets):
            return self.get_macro_scores_inputs_idx_targets_idx(tagger, inputs, targets)
        elif self.is_str(inputs) and self.is_str(targets):
            return self.get_macro_scores_tokens_tags(tagger, inputs, targets)
        else:
            raise ValueError('Unknown combination of inputs and targets')

    def get_macro_scores_inputs_tensor_targets_tensor(self, tagger, inputs_tensor, targets_tensor):
        targets_idx = self.sequences_indexer.tensor2idx(targets_tensor)
        return self.get_macro_scores_inputs_tensor_targets_idx(tagger, inputs_tensor, targets_idx)

    def get_macro_scores_inputs_idx_targets_idx(self, tagger, inputs_idx, targets_idx):
        inputs_tensor = self.sequences_indexer.idx2tensor(inputs_idx)
        return self.get_macro_scores_inputs_tensor_targets_idx(tagger, inputs_tensor, targets_idx)

    def get_macro_scores_tokens_tags(self, tagger, token_sequences, tag_sequences):
        inputs_idx = self.sequences_indexer.token2idx(token_sequences)
        targets_idx = self.sequences_indexer.tag2idx(tag_sequences)
        return self.get_macro_scores_inputs_idx_targets_idx(tagger, inputs_idx, targets_idx)

    def get_macro_f1_scores_details(self, tagger, token_sequences, tag_sequences):
        outputs_idx = tagger.predict_idx_from_tokens(token_sequences)
        targets_idx = self.sequences_indexer.tag2idx(tag_sequences)
        y_true = [i for sequence in targets_idx for i in sequence]
        y_pred = [i for sequence in outputs_idx for i in sequence]
        f1_scores = f1_score(y_true, y_pred, average=None)*100
        str = 'Tag    | MACRO-F1\n-----------------\n'
        for n in range(self.sequences_indexer.get_tags_num()):
            tag = self.sequences_indexer.idx2tag_dict[n+1]  # minumum tag no is "1"
            str += '%006s |  %1.2f\n' % (tag, f1_scores[n])
        str += '-----------------\n'
        str += '%006s |  %1.2f\n' % ('F1', np.mean(f1_scores))
        return str

    def write_report(self, fn, args, tagger, token_sequences, tag_sequences):

        text_file = open(fn, mode='w')
        for hyper_param in str(args).replace('Namespace(', '').replace(')', '').split(', '):
            text_file.write('%s\n' % hyper_param)

        acc_test, f1_test, precision_test, recall_test = self.get_macro_scores(tagger=tagger,
                                                                               inputs=token_sequences,
                                                                               targets=tag_sequences)

        text_file.write('\nResults on TEST: Accuracy = %1.2f, MACRO F1 = %1.2f, Precision = %1.2f, Recall = %1.2f.\n\n' % (
                                                             acc_test, f1_test, precision_test, recall_test))
        text_file.write(self.get_macro_f1_scores_details(tagger, token_sequences, tag_sequences))
        text_file.close()


    '''def get_macro_scores_inputs_tensor_targets_idx(self, tagger, inputs_tensor, targets_idx):
        outputs_idx = tagger.predict_idx_from_tensor(inputs_tensor)
        if len(targets_idx) != len(outputs_idx):
            raise ValueError('len(targets_idx) != len(len(outputs_idx))')
        num_data = len(targets_idx)
        accuracy_sum, f1_sum, precision_sum, recall_sum = 0, 0, 0, 0
        for n in range(num_data):
            accuracy_sum += accuracy_score(y_true=targets_idx[n], y_pred=outputs_idx[n])
            f1_sum += f1_score(y_true=targets_idx[n], y_pred=outputs_idx[n], average='macro')
            precision_sum += precision_score(y_true=targets_idx[n], y_pred=outputs_idx[n], average='macro')
            recall_sum += recall_score(y_true=targets_idx[n], y_pred=outputs_idx[n], average='macro')
        print("OLD style")
        return accuracy_sum / num_data, f1_sum / num_data, precision_sum / num_data, recall_sum / num_data'''