from const import COLORS, LINESTYLE, FIGSIZE, GEO_DIR
import matplotlib.pyplot as plt
import shapefile as shp

def plot_loss_auc(epochs: int, results: dict, title=None) -> None:

    labels = results['labels']
    losses = results['losses']
    aucs = results['aucs']
    assert len(labels) == len(losses) == len(aucs), 'wrong dims of inputs'
    epochs = list(range(0, epochs, epochs//len(losses[0])))

    _, axes = plt.subplots(2, 1, figsize=(20, 13))
    if title is None:
        title = f"{', '.join(labels)} comparison"

    axes[0].grid()
    axes[1].grid()
    axes[0].set_title(f'{title}, losses')
    axes[1].set_title(f'{title}, auc')

    for i, label in enumerate(labels):
        axes[0].plot(
            epochs, 
            losses[i], 
            color=COLORS[i], 
            label=f'{label} loss'
            )

        axes[1].plot(
            epochs, 
            aucs[i],
            color=COLORS[i], 
            linestyle=LINESTYLE, 
            label=f'{label} test auc')
    
    axes[0].legend()
    axes[1].legend()    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.show()

def plot_scatter_on_map(data: dict) -> None:

    sf = shp.Reader(f'{GEO_DIR}/rus_admbndl_ALL_2022_v01.shp')
    X, Y = [], []

    plt.figure(figsize=(20, 11))
    plt.title('Distribution of companies on the map of Russia')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        X.append(x)
        Y.append(y)
        plt.plot(x, y)

    labels = data['labels']
    lats = data['lats']
    lons = data['lons']
    assert len(labels) == len(lats) == len(lons), 'wrong dims of inputs'

    for i, label in enumerate(labels):
        plt.scatter(lons[i], lats[i], color=COLORS[i], label=label, s=8)
                
    plt.xlim(15, 185)
    plt.legend()
    plt.show()