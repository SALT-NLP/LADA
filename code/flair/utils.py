import random
def gen_knn_mix_batch(
	batch,
	train_dataset,
	sent_id_knn_array,
	knn_mix_ratio,
	train_size):
	"""
	we want to generate the mix bathc for knn data. 
	
	Inputs: 
	batch: the main training batch. a list of features. batch[5]: all_sent_id, a tensor of each sent's id
	train_dataset: the whole training set
	sent_id_knn_array: the I array from FAISS
	knn_mix_ratio: ratio of mix batch's samples are from the cluster, other samples are randomly selected. 
	train_size: the size of training set
	
	Outputs:
	mix_batch: similar to batch. to do mix
	"""
	
	#2. sample a sent_id for each sent 
	sent_id_mix_batch=[]

	for sent in batch:
		sent_id = sent.idx
		if random.uniform(0, 1) < knn_mix_ratio:
			sent_id_mix_batch.append(random.choice(sent_id_knn_array[sent_id]))
		else:
			sent_id_mix_batch.append(random.randint(0, train_size-1))
			
	#3. make the batch 
	mix_batch = [train_dataset[idx] for idx in sent_id_mix_batch]
	assert len(mix_batch)==len(batch),(len(mix_batch),len(batch))
	
	# print('sent_id_batch')
	# print(sent_id_batch)
	# print('sent_id_mix_batch')
	# print(sent_id_mix_batch)
	
	return mix_batch

