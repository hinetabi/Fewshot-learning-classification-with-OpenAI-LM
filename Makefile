train_base:
	python main.py --path_to_data dataset/few_shot

train_new_data: 
	python main.py --path_to_data dataset/fruit

# test_zero_shot: