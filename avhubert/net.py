import fairseq
import hubert_pretraining
import hubert
import hubert_asr
import fairseq
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, model_path, fix_encoder=False):
        super().__init__()
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])

        if hasattr(models[0], 'decoder'):
            print(f"Checkpoint: fine-tuned")
            self.backbone = models[0].encoder.w2v_model
        else:
            self.backbone = models[0]

        self.classifier = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        if fix_encoder:
            for param in self.backbone.parameters():
                param.require_grad = False

    def forward(self, video, audio, length):
        source = {'video': video, 'audio': audio}
        feature, _ = self.backbone.extract_finetune(source=source, padding_mask=None, output_layer=None)
        
        pooled_output = []
        for i in range(feature.shape[0]):
            pooled_output.append(feature[i, :length[i]].mean(dim=0))

        x = torch.stack(pooled_output)
        x = self.classifier(x)
        return x.squeeze(dim=1)