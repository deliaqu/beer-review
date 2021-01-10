Before running the code, please unzip the data and the embedding:
    
    1. unzip "review+wiki.filtered.200.txt.gz" to $EMBEDDING
    2. unzip "reviews.260k.train.txt.gz" to $TRAIN
    3. unzip "reviews.260k.heldout.txt.gz" to $DEV
    
Set $OUTPUT to be the path prefix that you want the output files to have.

To run the training, run the command:
    
    python3 training.py --train_data=$TRAIN --embedding=$EMBEDDING --dev_data=$DEV --output_prefix=$OUTPUT
    
Other controllable parameters that can be passed in as flags include
    
    --learning_rate
    --lambda_selection_cost
    --lambda_continuity_cost
    --hidden_dim_encoder
    --hidden_dim_generator
