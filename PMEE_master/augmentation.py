from config import *
"""
Implement sample classification function in the prototype model.
"""
def cat(features, labels, mode):
    if 'bert' in args.model or 'roberta' in args.model:
        feature = features[:, 0, :]
    elif 'xlnet' in args.model:
        feature = features[:, -1, :]
    if mode == 'cons':
        # If it is a contrastive learning stage, normalization is also required
        feature = feature.div(feature.pow(2).sum(1, keepdim=True).pow(1. / 2))  # L2 normalization
    cat_feature = [[], [], [], [], [], []]
    for i, label in enumerate(labels):
        #foe mosi
        # if label == -3.0:
        #     cat_feature[0].append(feature[i])
        # elif label == -2.0:
        #     cat_feature[1].append(feature[i])
        # elif label == -1.0:
        #     cat_feature[2].append(feature[i])
        # elif label == 0.0:
        #     cat_feature[3].append(feature[i])
        # elif label == 1.0:
        #     cat_feature[4].append(feature[i])
        # elif label == 2.0:
        #     cat_feature[5].append(feature[i])
        # elif label == 3.0:
        #     cat_feature[6].append(feature[i])
        #for mosei
        if label >= -3.0 and label < -2.0:
            cat_feature[0].append(feature[i])
        elif label >= -2.0 and label < -1.0:
            cat_feature[1].append(feature[i])
        elif label >= -1.0 and label < 0.0:
            cat_feature[2].append(feature[i])
        elif label >= 0.0 and label < 1.0:
            cat_feature[3].append(feature[i])
        elif label >= 1.0 and label < 2.0:
            cat_feature[4].append(feature[i])
        elif label >= 2.0 and label <= 3.0:
            cat_feature[5].append(feature[i])
    return cat_feature

def aug(features, labels, mode):
    #labels = np.round(labels.cpu())
    cat_feature = cat(features, labels, mode)
    if mode == 'aug':
        if 'base' in args.model:
            avg_cat_feature = torch.zeros([6, 768]).to(DEVICE)
        elif 'large' in args.model:
            avg_cat_feature = torch.zeros([6, 1024]).to(DEVICE)
        for i, fe in enumerate(cat_feature):
            if len(fe) > 0:
                avg_cat_feature[i] = avg_cat_feature[i] + sum(fe) / len(fe)

        return avg_cat_feature
    else:
        x = -3
        new_features, new_labels = [], []
        for index, i in enumerate(cat_feature):
            if len(i) > 1:
                new_features.extend(i)
                for j in range(len(i)):
                    new_labels.append(x)
            x += 1
        featuress = torch.tensor([item.cpu().detach().numpy() for item in new_features]).cuda()
        return featuress, torch.Tensor(new_labels)



