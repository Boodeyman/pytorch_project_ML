import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate function для ASVspoof dataset с поддержкой спектрограмм
    """
    result_batch = {}

    # Проверяем, что у нас есть - data_object (синтетические данные) или спектрограммы
    if "data_object" in dataset_items[0]:
        # Оригинальная логика для синтетических данных
        result_batch["data_object"] = torch.stack(
            [item["data_object"] for item in dataset_items]
        )
    elif "spectrogram" in dataset_items[0]:
        # Логика для спектрограмм
        spectrograms = [item["spectrogram"] for item in dataset_items]

        # Паддинг спектрограмм до одинакового размера по времени
        if len(spectrograms) > 0:
            max_time = max(spec.size(-1) for spec in spectrograms)

            padded_specs = []
            for spec in spectrograms:
                current_time = spec.size(-1)

                if current_time < max_time:
                    # Паддинг по времени
                    pad_time = max_time - current_time
                    spec = torch.nn.functional.pad(
                        spec, (0, pad_time), mode="constant", value=0
                    )

                padded_specs.append(spec)

            result_batch["data_object"] = torch.stack(padded_specs, dim=0)

    # Метки одинаковы в любом случае
    result_batch["labels"] = torch.tensor(
        [item["labels"] for item in dataset_items], dtype=torch.long
    )

    return result_batch
