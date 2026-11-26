from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "small_regions": {"file": "small_regions.csv", "name": "Область",
                      "features": ["Машиностроение %", "Электроэнергетика %"], "target": None},
    "regions": {"file": "regions.csv", "name": "Название",
                "features": ["Покупка товаров и оплата услуг", "Приобретение недвижимости",
                             "Прирост (уменьшение) денег на руках у населения"], "target": None},
    "wine": {"file": "wine.csv", "name": None,
             "features": ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
                          "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
                          "color_intensity", "hue", "od280_od315_of_diluted_wines", "proline"],
             "target": "class"}
}

def minkowski_distance(x, y, p=2):
    """Расстояние Минковского между двумя векторами."""
    if p < 1:
        raise ValueError("Для метрики Минковского p должно быть >= 1.")
    return float(np.sum(np.abs(np.asarray(x) - np.asarray(y)) ** p) ** (1.0 / p))

def initial_distance_matrix(X, p=2):
    """Самостоятельно строит симметричную матрицу попарных расстояний."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
        raise ValueError("X должна быть конечной числовой матрицей минимум из двух объектов.")
    if p < 1:
        raise ValueError("p должно быть >= 1.")
    diff = np.abs(X[:, None, :] - X[None, :, :])
    D = np.sum(diff ** p, axis=2) ** (1.0 / p)
    np.fill_diagonal(D, 0.0)
    return D

def manual_complete_linkage(X, p=2, trace=False):
    """
    Самостоятельная агломеративная кластеризация методом дальнего соседа.
    Complete linkage: D(A∪B,C) = max(D(A,C), D(B,C)).
    Возвращает linkage-матрицу формата SciPy: [id_A, id_B, height, size].
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if n < 2:
        raise ValueError("Для кластеризации нужны минимум два объекта.")
    D0 = initial_distance_matrix(X, p=p)
    # Максимум 2*n-1 узлов: исходные объекты и создаваемые кластеры.
    D = np.zeros((2*n - 1, 2*n - 1), dtype=float)
    D[:n, :n] = D0
    active = list(range(n))
    sizes = {i: 1 for i in range(n)}
    history = []

    if trace:
        print_matrix(D0, list(range(n)), "R0 — исходная матрица")

    while len(active) > 1:
        current = D[np.ix_(active, active)].copy()
        np.fill_diagonal(current, np.inf)
        flat = int(np.argmin(current))
        i, j = np.unravel_index(flat, current.shape)
        if i > j:
            i, j = j, i
        a, b = active[i], active[j]
        height = float(D[a, b])
        new_id = n + len(history)
        new_size = sizes[a] + sizes[b]

        # Complete linkage: расстояние до нового кластера — максимум двух старых.
        remaining = [node for pos, node in enumerate(active) if pos not in (i, j)]
        for other in remaining:
            D[new_id, other] = D[other, new_id] = max(D[a, other], D[b, other])

        history.append([a, b, height, new_size])
        if trace:
            print(f"\nСлияние C{a} + C{b} -> C{new_id}; d={height:.6f}; "
                  f"размер={new_size}")
            if remaining:
                other = remaining[0]
                print(f"До C{other}: max({D[a, other]:.6f}, {D[b, other]:.6f}) "
                      f"= {D[new_id, other]:.6f}")
        active = [node for pos, node in enumerate(active) if pos not in (i, j)]
        active.append(new_id)
        sizes[new_id] = new_size
        if trace and len(active) <= 20:
            active_matrix = D[np.ix_(active, active)]
            print_matrix(active_matrix, active, f"Матрица после слияния {len(history)}")

    return np.asarray(history, dtype=float)

def print_matrix(matrix, ids, title):
    print(f"\n{title}")
    frame = pd.DataFrame(matrix, index=[f"C{i}" for i in ids],
                         columns=[f"C{i}" for i in ids])
    print(frame.round(3).to_string())

def labels_at_k(history, n, k):
    """Восстанавливает разбиение после первых n-k слияний."""
    if not 1 <= k <= n:
        raise ValueError(f"k должно быть от 1 до {n}.")
    groups = {i: [i] for i in range(n)}
    for step, row in enumerate(history[:n-k]):
        a, b = int(row[0]), int(row[1])
        groups[n + step] = groups.pop(a) + groups.pop(b)
    labels = np.empty(n, dtype=int)
    ordered = sorted(groups.values(), key=min)
    for label, members in enumerate(ordered, start=1):
        labels[members] = label
    return labels

def load_dataset(dataset):
    cfg = DATASETS[dataset]
    df = pd.read_csv(ROOT / cfg["file"])
    required = cfg["features"] + ([cfg["target"]] if cfg["target"] else [])
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"В {cfg['file']} отсутствуют столбцы: {missing_columns}")
    before = len(df)
    duplicate_count = int(df.duplicated().sum())
    missing_count = int(df[required].isna().any(axis=1).sum())
    df = df.drop_duplicates().dropna(subset=required).copy()
    X_df = df[cfg["features"]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(X_df.to_numpy(dtype=float)).all(axis=1)
    df = df.loc[valid].copy()
    X_df = X_df.loc[valid]
    if len(df) < 2:
        raise ValueError("После очистки осталось меньше двух объектов.")
    names = (df[cfg["name"]].astype(str).tolist() if cfg["name"]
             else [f"Объект {i+1}" for i in range(len(df))])
    target = df[cfg["target"]].to_numpy() if cfg["target"] else None
    print(f"\nДатасет: {dataset}")
    print(f"Строк до очистки: {before}; после очистки: {len(df)}")
    print(f"Полные дубликаты: {duplicate_count}; строки с пропусками: {missing_count}")
    print(f"Признаков: {len(cfg['features'])}")
    print("Признаки:", ", ".join(cfg["features"]))
    print("Диапазоны признаков:")
    print(pd.DataFrame({"min": X_df.min(), "max": X_df.max(),
                        "mean": X_df.mean()}).round(4).to_string())
    return X_df.to_numpy(dtype=float), names, target, cfg["features"]

def scale_features(X, mode):
    X = np.asarray(X, dtype=float)
    if mode == "none":
        return X.copy()
    if mode == "minmax":
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = hi - lo
        span[span == 0] = 1.0
        return (X - lo) / span
    if mode == "standard":
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0
        return (X - mean) / std
    raise ValueError("Неизвестный способ масштабирования.")

def plot_experiment(dataset, X, labels, history, p, scale, out_dir, show=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    dendrogram(history, ax=axes[0], truncate_mode="lastp" if len(X) > 80 else None,
               p=30 if len(X) > 80 else 0, leaf_rotation=60, leaf_font_size=8)
    axes[0].set_title("Дендрограмма — Complete Linkage")
    axes[0].set_xlabel("Объекты / свёрнутые ветви")
    axes[0].set_ylabel("Расстояние Минковского")
    if X.shape[1] > 2:
        coords = PCA(n_components=2).fit_transform(X)
        xlab, ylab = "Первая компонента PCA", "Вторая компонента PCA"
    else:
        coords = X[:, :2]
        xlab, ylab = "Признак 1", "Признак 2"
    for label in np.unique(labels):
        mask = labels == label
        axes[1].scatter(coords[mask, 0], coords[mask, 1], label=f"Кластер {label} (n={mask.sum()})",
                        alpha=0.8)
    axes[1].set_title("Кластеры в 2D-проекции")
    axes[1].set_xlabel(xlab)
    axes[1].set_ylabel(ylab)
    axes[1].legend()
    fig.suptitle(f"{dataset}: p={p}, масштаб={scale}, k={len(np.unique(labels))}")
    fig.savefig(out_dir / f"{dataset}_p{p}_{scale}.png", dpi=160)
    if show:
        plt.show()
    plt.close(fig)

def run_experiment(dataset, p=2, k=2, scale="minmax", output=None, trace=False, show=False):
    X_raw, names, target, feature_names = load_dataset(dataset)
    X = scale_features(X_raw, scale)
    n = len(X)
    if not 1 <= k <= n:
        raise ValueError(f"k должно быть от 1 до {n}.")
    print(f"\nПараметры: p={p}; linkage=complete (дальний сосед); scale={scale}; k={k}")
    manual = manual_complete_linkage(X, p=p, trace=trace)
    scipy_tree = linkage(pdist(X, metric="minkowski", p=p), method="complete")
    manual_labels = labels_at_k(manual, n, k)
    scipy_labels = labels_at_k(scipy_tree, n, k)
    ari = adjusted_rand_score(manual_labels, scipy_labels)
    height_error = float(np.max(np.abs(np.sort(manual[:, 2]) - np.sort(scipy_tree[:, 2]))))
    sizes = np.bincount(manual_labels)[1:].tolist()
    print(f"\nСлияний до k={k}: {n-k}")
    print("Размеры кластеров (ручной алгоритм):", sizes)
    print(f"ARI ручной алгоритм / SciPy: {ari:.6f}")
    print(f"Максимальная разница отсортированных высот: {height_error:.12g}")
    if target is not None:
        print(f"ARI кластеров / исходный класс (только внешняя проверка): "
              f"{adjusted_rand_score(target, manual_labels):.6f}")
    result = pd.DataFrame({"Объект": names, "Кластер": manual_labels})
    if target is not None:
        result["Исходный класс (не использовался при обучении)"] = target
    out_dir = Path(output) if output else ROOT / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / f"{dataset}_clusters_p{p}_{scale}.csv", index=False)
    merges = pd.DataFrame(manual, columns=["cluster_a", "cluster_b", "distance", "size"])
    merges.insert(0, "step", np.arange(1, len(merges)+1))
    merges.to_csv(out_dir / f"{dataset}_merges_p{p}_{scale}.csv", index=False)
    plot_experiment(dataset, X, manual_labels, manual, p, scale, out_dir, show=show)
    return {"dataset": dataset, "n": n, "features": len(feature_names), "p": p,
            "scale": scale, "k": k, "cluster_sizes": sizes, "ari_scipy": ari,
            "max_height_difference": height_error}

def menu():
    mapping = {"1": "small_regions", "2": "regions", "3": "wine"}
    while True:
        print("\nВыберите датасет:")
        print("1. small_regions.csv — 6 областей (обязательный)")
        print("2. regions.csv — 18 регионов (обязательный)")
        print("3. wine.csv — 178 образцов вина, 13 признаков (собственный)")
        print("0. Выход")
        choice = input("Ваш выбор: ").strip()
        if choice == "0":
            break
        if choice not in mapping:
            print("Введите 1, 2, 3 или 0.")
            continue
        try:
            p = float(input("Параметр Минковского p [2]: ").strip() or "2")
            k = int(input("Количество кластеров k [2]: ").strip() or "2")
            scale = input("Масштабирование none/minmax/standard [minmax]: ").strip() or "minmax"
            run_experiment(mapping[choice], p=p, k=k, scale=scale,
                           trace=(choice == "1"), show=True)
        except (ValueError, OSError) as exc:
            print("Ошибка:", exc)

def main():
    parser = argparse.ArgumentParser(description="Агломеративная кластеризация: Минковский + дальний сосед")
    parser.add_argument("--dataset", choices=list(DATASETS))
    parser.add_argument("--p", type=float, default=2.0, help="Параметр Минковского, p >= 1")
    parser.add_argument("--k", type=int, default=2, help="Число кластеров")
    parser.add_argument("--scale", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--output", type=Path, default=ROOT / "graphs")
    parser.add_argument("--trace", action="store_true", help="Печатать матрицы и шаги слияния")
    parser.add_argument("--show", action="store_true", help="Открывать графики")
    args = parser.parse_args()
    if args.p < 1:
        parser.error("--p должен быть >= 1")
    if args.dataset is None:
        menu()
    else:
        run_experiment(args.dataset, args.p, args.k, args.scale, args.output,
                       trace=args.trace, show=args.show)

if __name__ == "__main__":
    main()
