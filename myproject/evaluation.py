from mxboard import SummaryWriter
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
                
class Evaluator():
    def __init__(self, experiment_id, prefix, console=True, mxboard=False):
        """
        :param experiment_id: experiment identifier
        :type experiment_id: str
        :param prefix: used as a prefix in all metric labels
        :type prefix: str
        :param console: flag for console logging of metrics
        :type console: bool
        :param mxboard: flag for mxboard logging of metrics
        :type mxboard: bool
        """
        self.reset()
        self._experiment_id = experiment_id
        self._prefix = prefix
        self._console = console
        self._mxboard = mxboard
    
    def update(self, label, pred, pred_proba, loss):
        assert label.shape[0] == 1, "Only batch_size of 1 supported"
        label = label.asnumpy().flatten()
        self._labels.append(label)  
        pred = pred.asnumpy().flatten()
        self._preds.append(pred)
        pred_proba = pred_proba.asnumpy().flatten()
        self._pred_probs.append(pred_proba)
        self._losses.append(loss.asnumpy())
    
    def reset(self):
        self._labels = AppendableArray()
        self._preds = AppendableArray()
        self._pred_probs = AppendableArray()
        self._losses = AppendableArray()
        
    def get(self):
        return self._labels.get(), self._preds.get(), self._pred_probs.get(), self._losses.get()
    
    def loss(self):
        return self._losses.get().mean()
    
    def accuracy(self):
        return accuracy_score(self._labels.get(), self._preds.get())
    
    def roc_auc(self):
        return roc_auc_score(self._labels.get(), self._pred_probs.get())
    
    def precision(self):
        return precision_score(self._labels.get(), self._preds.get())
    
    def recall(self):
        return recall_score(self._labels.get(), self._preds.get())
    
    def f1(self):
        return f1_score(self._labels.get(), self._preds.get())
    
    def average_precision(self):
        return average_precision_score(self._labels.get(), self._pred_probs.get())
 
    def window_diff(self):
        labels = self._labels.get(concat=False)
        preds = self._preds.get(concat=False)
        diffs = []
        for label, pred in zip(labels, preds):
            diff = window_diff(pred=pred, label=label)
            diffs.append(diff)
        avg_diff = np.array(diffs).mean()
        return avg_diff

    def log(self, step_idx):
        if self._console:
            self.log_console()
        if self._mxboard:
            self.log_mxboard(step_idx)
        
    def plot(self, sample_id, step_idx, pred_proba, pred, label):
        if self._console:
            self.plot_console(sample_id, step_idx, pred_proba, pred, label)
        if self._mxboard:
            self.plot_mxboard(sample_id, step_idx, pred_proba, pred, label)
            
    def plot_console(self, sample_id, step_idx, pred_proba, pred, label):
        pass
    
    def plot_mxboard(self, sample_id, step_idx, pred_proba, pred, label):
        pred_proba = pred_proba[0].asnumpy()
        pred = pred[0].asnumpy()
        label = label.asnumpy()
        img_array = plot_segment_breaks(pred_proba, pred, label)
        img_array = img_array.transpose((2,0,1))[:3].astype(np.float32)/255
        with SummaryWriter(logdir="../logs/" + self._experiment_id) as sw:
            sw.add_image(tag=sample_id, image=img_array, global_step=step_idx)

    def log_console(self):
        t = PrettyTable(['Metric', 'Value'])
        t.add_row(['loss', round(self.loss(), 5)])
        t.add_row(['accuracy', round(self.accuracy(), 5)])
        t.add_row(['roc_auc', round(self.roc_auc(), 5)])
        t.add_row(['precision', round(self.precision(), 5)])
        t.add_row(['recall', round(self.recall(), 5)])
        t.add_row(['f1', round(self.f1(), 5)])
        t.add_row(['average_precision', round(self.average_precision(), 5)])
        t.add_row(['window_diff', round(self.window_diff(), 5)])
        print(t)
    
    def log_mxboard(self, step_idx):
        with SummaryWriter(logdir="../logs/" + self._experiment_id) as sw:
            ### scalars
            sw.add_scalar(tag=self._prefix + '_loss',
                          value=self.loss(), global_step=step_idx)
            sw.add_scalar(tag=self._prefix + '_accuracy',
                          value=self.accuracy(), global_step=step_idx)
            sw.add_scalar(tag=self._prefix + '_roc_auc',
                          value=self.roc_auc(), global_step=step_idx)
            sw.add_scalar(tag=self._prefix + '_precision',
                          value=self.precision(), global_step=step_idx)
            sw.add_scalar(tag=self._prefix + '_recall',
                          value=self.recall(), global_step=step_idx)
            sw.add_scalar(tag=self._prefix + '_f1',
                          value=self.f1(), global_step=step_idx)
            sw.add_scalar(tag=self._prefix + '_average_precision',
                          value=self.average_precision(), global_step=step_idx)
            sw.add_scalar(tag=self._prefix + '_window_diff',
                          value=self.window_diff(), global_step=step_idx)
            ### histograms
            sw.add_histogram(tag=self._prefix + '_loss_histogram',
                             values=self._losses.get(), bins=100, global_step=step_idx)
            ### other
            sw.add_pr_curve(tag=self._prefix + '_precision_recall',
                            labels=self._labels.get(), predictions=self._pred_probs.get(),
                            num_thresholds=100, global_step=step_idx)