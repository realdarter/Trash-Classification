from image_classifier import *  # Ensure this imports the necessary functions and classes

if __name__ == '__main__':
    data_dir = 'data/garbage_classification'
    model_dir = 'saved_models'
    
    # Training args
    args = create_args(
        num_epochs=1, 
        batch_size=32, 
        learning_rate=1e-5, 
        save_every=500, 
        max_length=256, 
        temperature=1.0, 
        top_k=5, 
        top_p=0.9, 
        repetition_penalty=1.0
    )
    
    # Call the train function
    train(data_dir, model_dir, args)