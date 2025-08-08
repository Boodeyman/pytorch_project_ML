import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxFeatureMap2D(nn.Module):
    """Max Feature Map activation для LCNN"""

    def __init__(self, max_dim=1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, x):
        # Разделяем каналы на пары и берем максимум
        s1, s2 = torch.split(x, x.size(self.max_dim) // 2, dim=self.max_dim)
        return torch.max(s1, s2)


class LCNNBlock(nn.Module):
    """Базовый блок LCNN с batch normalization"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.mfm = MaxFeatureMap2D(max_dim=1)
        self.bn = nn.BatchNorm2d(out_channels // 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.mfm(x)
        x = self.bn(x)
        return x


class BaselineModel(nn.Module):
    """
    LCNN для Voice Anti-spoofing (заменяет BaselineModel)
    """

    def __init__(self, n_feats=257, n_class=2, dropout=0.75):
        """
        Args:
            n_feats (int): размерность частот в спектрограмме (n_fft//2 + 1).
            n_class (int): количество классов (2 для bonafide/spoof).
            dropout (float): коэффициент dropout.
        """
        super().__init__()

        # Feature extraction layers согласно STC paper
        self.conv1 = LCNNBlock(1, 64, (5, 5), stride=(1, 1), padding=(2, 2))
        self.pool1 = nn.MaxPool2d((2, 2))
        self.bn_pool1 = nn.BatchNorm2d(32)

        self.conv2 = LCNNBlock(32, 64, (1, 1), stride=(1, 1))
        self.conv3 = LCNNBlock(32, 96, (3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d((2, 2))
        self.bn_pool2 = nn.BatchNorm2d(48)

        self.conv4 = LCNNBlock(48, 96, (1, 1), stride=(1, 1))
        self.conv5 = LCNNBlock(48, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.pool3 = nn.MaxPool2d((2, 2))
        self.bn_pool3 = nn.BatchNorm2d(64)

        self.conv6 = LCNNBlock(64, 128, (1, 1), stride=(1, 1))
        self.conv7 = LCNNBlock(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv8 = LCNNBlock(32, 64, (1, 1), stride=(1, 1))
        self.conv9 = LCNNBlock(32, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.pool4 = nn.MaxPool2d((2, 2))
        self.bn_pool4 = nn.BatchNorm2d(32)

        # Global pooling для разных размеров входа
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, n_class)

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, data_object, **batch):
        """
        Args:
            data_object (Tensor): спектрограмма [B, F, T] или вектор [B, features]
        """
        x = data_object

        # Если 3D спектрограмма [B, F, T], добавляем канал
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, F, T]

        # Если 2D синтетические данные [B, features]
        elif x.dim() == 2:
            # Преобразуем [B, 1024] -> [B, 1, 32, 32] для CNN
            batch_size = x.size(0)
            feature_size = x.size(1)
            sqrt_size = int(feature_size**0.5)
            if sqrt_size * sqrt_size != feature_size:
                target_size = 32 * 32
                if feature_size < target_size:
                    x = torch.nn.functional.pad(x, (0, target_size - feature_size))
                else:
                    x = x[:, :target_size]
                sqrt_size = 32
            x = x.view(batch_size, 1, sqrt_size, sqrt_size)

        # Остальной код без изменений
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn_pool1(x)

        # ... остальные блоки ...
        # (весь остальной код остается как был)

        # Block 2
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.bn_pool2(x)

        # Block 3
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.bn_pool3(x)

        # Block 4
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.pool4(x)
        x = self.bn_pool4(x)

        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        # Classification
        x = self.dropout(x)
        logits = self.fc(x)

        return {"logits": logits}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
