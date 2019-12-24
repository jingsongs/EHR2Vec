# MedE2Vec
MedE2Vec is a embedding translation tool for medical entities based on attention mechanism. We have provided a version of tensorflow implement and it is constantly being improved.

# Prerequisites
1.	We use Python 3.6 and Tensorflow 1.8.0. 
2.	Download/clone the MedE2Vec code


# Running
1.	Preparing training data:
  Training data should be a list of list of medical entities include diagnosis, medication, labtest and symptom and the medical entities should be transformed to integers. In our model, a patient medical event includes many medical entities and this medical events of a patient constitute a patient complete medical process.In the meantime,a patient medical events are sorted in chronological order. So an out layer list denote a patient, each inner layer denote each event of this patient and we use -1 to separate every event of a patient. For example,  [[22,50,33], [-1] ,[4,58,60],[20]] means there are two patients where the first patient only had one event which include three entities and the second patient had two event include [4,58,60] and [20]. And you should save the transformation vocabulary (from medical entities to integers). Both of the two files need to be pickled us python pickle.

2.	Model parameters configuration:
   The max length of the medical event of all patients needs to be count and called maxlen_seqs in our model. The number of all your medical entities in your vocabulary is required in our model and called n_input. The default dimension of entity is 512 which you can change as you like. The attention mechanism parameters include num_heads,num_blokcs and dff you can set. The number of epoch and the size of batch should be configured for your own machine.

3.	Running:
      After the step of configuration, all of the parameters are set in the configuration dictionary. Don’t forget the training data path and output path. You can use the simple execution command: python3 SLE_MedE2Vec_Runner.py to run the model.

## Example of how to run MedE2vec with the provide train_data_example.pkl and dict_types.pkl
1、	Count the total number input entities(n_input) and the max length(max_seq_length) of all events.
                          python3 statistic.py
       
2、	Put the (n_input+1) and max_seq_length parameter into the get_config() function.

3、	Put your save model path into the get_config() function.

4、	The default data_path and dict_types_path of this example were configured and you can set it for yourself.

5、	Run the MedE2vec model and get the entity embedding:
                        python3 SLE_MedE2Vec_Runner.py
       
6、The vectors result can be found in the model path.
