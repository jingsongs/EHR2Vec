# MedE2Vec
MedE2Vec is an embedding translation tool for medical entities based on attention mechanism. We have provided a version of tensorflow implement and it is constantly being improved.

# Prerequisites
1.	We use Python 3.6 and Tensorflow 1.8.0. 
2.	Download/clone the MedE2Vec code


# Running
1.	Preparing training data:
  Training data should be a list of list of medical entities include diagnosis, medication, labtest and symptom and the medical entities should be transformed to integers. In our model, a patient medical event includes many medical entities and this medical events of a patient constitute a patient complete medical process.In the meantime,a patient medical events are sorted in chronological order. So an out layer list denote a patient, each inner layer denote each event of this patient and we use -1 to separate every event of a patient. For example,  [[22,50,33], [-1] ,[4,58,60],[20]] means there are two patients where the first patient only had one event which include three entities and the second patient had two event include [4,58,60] and [20]. And you should save the transformation vocabulary (from medical entities to integers). Both of the two files need to be pickled us python pickle.

2.	Model hyper-parameters configuration:
   The max length of the medical event of all patients needs to be count and called maxlen_seqs in our model. The number of all your medical entities in your vocabulary is required in our model and called n_input. The default dimension of entity is 512 which you can change as you like. The attention mechanism parameters include num_heads,num_blokcs and dff you can set. The number of epoch and the size of batch should be configured for your own machine.

3.	Running:
You can train the model with the default hyper-parameters except the training data path,n_input,maxlen_seqs, dict_types_path,save_model_path and embedding save path. And you can use the simple execution command  to run the model: 

    python3 SLE_MedE2Vec_Runner.py --data_path  <your train data path>  --n_input <the entity number of your data>  --maxlen_seqs  <the max event length of your data>  --dict_types_path <your vocabulary path> --entity_embedding_path  <save embedding path>  --save_model_path  <save your model path>

## Example of how to run MedE2Vec with the provide train_data_example.pkl and dict_types.pkl

1、	Count the total number of input entities(n_input) and the max length(max_seq_length) of all events with the command.

     python3 statistic.py --dict_types ./example/dict_types.pkl --example_train_path ./example/train_data_example.pkl
    
    You will get the n_input number and max_seq_length number from the output.
2、	The n_input number needs to plus one for the model.

3、	Run the MedE2Vec model and get the entity embedding:

    python3 SLE_MedE2Vec_Runner.py --n_input 17 --maxlen_seqs 7 --data_path ./example/train_data_example.pkl --dict_types_path ./example/dict_types.pkl --save_model_path ./example/MedE/ --entity_embedding_path ./example/entity_embedding.pkl

4、The vectors result can be found in the entity_embedding_path.


