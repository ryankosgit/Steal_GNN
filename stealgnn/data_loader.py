"""**LOAD DATA & INFO**"""

dataset_name = "" #cora, ogb-arxiv, pubmed, computers

attack_type = "" #1, 2, 3
in_dim = 0
hid_dim = 0
num_classes = 0

if dataset_name == 'cora':
    dataset = Planetoid(root='./data', name='Cora')
    in_dim = 1433; hid_dim = 128; num_classes = 7

elif dataset_name == 'ogb-arxiv':
    torch.serialization.add_safe_globals([DataEdgeAttr])
    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./data')
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    train_idx, test_idx = split_idx['train'], split_idx['test']
    x = data.x
    y = data.y.squeeze()
    edge_index = data.edge_index
    in_dim = 128; hid_dim = 256; num_classes = 40; num_layers = 3

elif dataset_name == 'pubmed':
    dataset = Planetoid(root='./data', name='Pubmed')
    in_dim = 500; hid_dim = 128; num_classes = 3

elif dataset_name == 'computers':
    dataset = Amazon(root='data/Amazon', name='computers', transform=NormalizeFeatures())
    data = dataset[0]
    data = RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)(data)

    in_dim = data.x.shape[1]
    hid_dim = 128
    num_classes = int(data.y.max().item()) + 1

    print(f"Max label: {data.y.max().item()}, Num classes: {num_classes}")
    assert data.y.max().item() < num_classes, "Invalid label value in data.y"



data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

"""**ATTACK INFO + CLASSES**"""
