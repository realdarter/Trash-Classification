from image_classifier import *  # Ensure this imports the necessary functions and classes

data_dir = 'data/garbage_classification'
model_dir = 'saved_models/save1'

args = create_args(
    num_epochs=2, 
    batch_size=32, 
    learning_rate=1e-5, 
    save_every=500, 
    max_length=256, 
    temperature=1.0, 
    top_k=5, 
    top_p=0.9, 
    repetition_penalty=1.0
    )
    
history = train(data_dir=data_dir, model_dir=model_dir, args=args)
print(history)