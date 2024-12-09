from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
import os
import pandas as pd

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    # Add static counter for test flag
    if not hasattr(data_provider, 'test_printed'):
        data_provider.test_printed = False

    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )

    # Create data loader
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    # Skip the rest if it's a test flag and we've already printed it
    if flag == 'test' and data_provider.test_printed:
        return data_set, data_loader

    # Get file size
    file_path = os.path.join(args.root_path, args.data_path)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB

    # Read the original dataframe to get dates
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    # Calculate date ranges based on the same splits used in Dataset_Custom
    num_total = len(df)
    num_train = int(num_total * Dataset_Custom.TRAIN_SPLIT)
    num_test = int(num_total * Dataset_Custom.TEST_SPLIT)
    num_val = num_total - num_train - num_test

    train_dates = df['date'].iloc[:num_train]
    val_dates = df['date'].iloc[num_train:num_train + num_val]
    test_dates = df['date'].iloc[-num_test:]

    # Map flag to dataset info
    dataset_info = {
        'train': {
            'dates': train_dates,
            'prefix': f"Dataset: {args.data_path} ({file_size:.1f}MB)\nTrain"
        },
        'val': {
            'dates': val_dates,
            'prefix': "Vali"
        },
        'test': {
            'dates': test_dates,
            'prefix': "Test"
        }
    }

    if flag in dataset_info:
        info = dataset_info[flag]
        years = (info['dates'].max() - info['dates'].min()).days / 365.25
        
        # Format components with fixed widths
        samples_str = f"{len(data_set):<3} time steps"
        years_str = f"~{years:.1f} years"
        dates_str = f"start date {info['dates'].min().strftime('%Y-%m-%d')}, end date {info['dates'].max().strftime('%Y-%m-%d')}"
        
        # Construct message differently for train vs other flags
        if flag == 'train':
            msg = f"\nDataset: {args.data_path} ({file_size:.1f}MB)\n"
            msg += f"Train: {samples_str:<13}, {years_str:<11}, {dates_str}\n"
        else:
            msg = f"{info['prefix']}:  {samples_str:<13}, {years_str:<11}, {dates_str}\n"

        # Write to both console and log file
        with open('/dev/tty', 'w') as console:
            console.write(msg)
        print(msg, end='')  # Log file output only

        # Mark test as printed if this was the test flag
        if flag == 'test':
            data_provider.test_printed = True

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
