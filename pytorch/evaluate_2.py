from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score
from pytorch_utils import forward
import numpy

class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        
    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)
        




        fileter = numpy.zeros((target.shape[0],target.shape[1]))
        for i in numpy.argmax(target, axis=1):
            #print(i)
            #print(f)
            fileter[:,i]=1
        clipwise_output = fileter * clipwise_output






        print('=====================')
        #print(target.shape)
        #print(numpy.sum(target, axis=1))
        #print(output_dict)
        #print(target.shape)
        #print(numpy.argmax(target, axis=1))
        #print(numpy.argmax(clipwise_output, axis=1))
        print('ACC')
        acc= accuracy_score(numpy.argmax(clipwise_output, axis=1),numpy.argmax(target, axis=1))
        print(acc)
        print('Precision')
        pre = precision_score(numpy.argmax(target, axis=1),numpy.argmax(clipwise_output, axis=1),average='macro')
        recall = recall_score(numpy.argmax(target, axis=1),numpy.argmax(clipwise_output, axis=1), average='macro')
        print(pre)
        print('Recall')
        print(recall)
        print('F1 score')
        f1 = f1_score(numpy.argmax(target, axis=1),numpy.argmax(clipwise_output, axis=1), average='macro')
        print(f1)

        #average_precision = metrics.average_precision_score(
        #    target, clipwise_output, average=None)
        
        #auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        
        statistics = {'f1_score': f1}#, 'auc': auc}

        return statistics
