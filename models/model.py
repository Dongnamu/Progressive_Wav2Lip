from torch import nn
from torch.functional import norm
from torch.nn import functional as F
import torch
from torch.nn.modules import normalization
from torch.nn.modules.container import Sequential

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,normalize=True, leakyRelu=None, residual=False):
        super().__init__()

        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
            # layers.append(nn.InstanceNorm2d(out_channels))
        
        if not leakyRelu:
            self.act = nn.ReLU()
        else:
            self.act = nn.LeakyReLU(leakyRelu)
        
        self.residual = residual

        self.down = nn.Sequential(*layers)
    def forward(self, x):
        out = self.down(x)

        if self.residual:
            out += x
        
        out = self.act(out)
        
        return out

class Conv2dTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, normalize=True, leakyRelu=None):
        super().__init__()

        layers = [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if not leakyRelu:
            self.act = nn.ReLU()
        else:
            self.act = nn.LeakyReLU(leakyRelu)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return self.act(out)
    
class faceProgressionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            Conv2d(out_channels, out_channels, kernel_size, 1, padding, normalize=False, residual=True),
            Conv2d(out_channels, out_channels, kernel_size, 1, padding, normalize=False, residual=True)
        )

    def forward(self, x):
        return self.block(x)

class faceDecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0):
        super().__init__()

        self.block = nn.Sequential(
            Conv2dTranspose(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            Conv2d(out_channels, out_channels, kernel_size, 1, padding, normalize=False, residual=True),
            Conv2d(out_channels, out_channels, kernel_size, 1, padding, normalize=False, residual=True)
        )

    def forward(self, x):
        return self.block(x)

class PSyncNet_color(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.progression = nn.ModuleList([
            faceProgressionBlock(32,64, stride=2), # unused
            faceProgressionBlock(64,128, stride=2), # 128, 128
            faceProgressionBlock(128,256, stride=2), # 64, 64
            faceProgressionBlock(256,512, stride=2), # 32, 32
            faceProgressionBlock(512,512, stride=2), # 16, 16
            faceProgressionBlock(512,512, stride=2), # 8, 8
            faceProgressionBlock(512,512, stride=4) # 4, 4
        ])

        self.fromRGB  = nn.ModuleList([
            Conv2d(15, 32, 1, 1, 0), # unused
            Conv2d(15, 64, 1, 1, 0),
            Conv2d(15, 128, 1, 1, 0), 
            Conv2d(15, 256, 1, 1, 0), 
            Conv2d(15, 512, 1, 1, 0),
            Conv2d(15, 512, 1, 1, 0), 
            Conv2d(15, 512, 1, 1, 0),
        ])
        
        self.n_layers = len(self.progression)

    def forward(self, audio_sequences, face_sequences, step=0):
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        
        for i in range(step, -1, -1):
            index = self.n_layers - i - 1

            if i == step:
                face_embedding = self.fromRGB[index](face_sequences)
            
            face_embedding = self.progression[index](face_embedding)

        # face_embedding = face_embedding.squeeze(2).squeeze(2)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding

class PWav2Lip(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        
        self.face_encoder = nn.ModuleList([
            faceProgressionBlock(64, 128, stride=2), # 128, 128
            faceProgressionBlock(128, 256, stride=2), # 64, 64
            faceProgressionBlock(256, 512, stride=2), # 32, 32
            faceProgressionBlock(512, 512, stride=2), # 16, 16
            faceProgressionBlock(512, 512, stride=2), # 8, 8
            faceProgressionBlock(512, 512, stride=4) # 4, 4
        ])

        self.from6to16 = Conv2d(6, 16, 7, 1, 3)

        self.from16 = nn.ModuleList([
            Conv2d(16, 64, 1, 1, 0),
            Conv2d(16, 128, 1, 1, 0),
            Conv2d(16, 256, 1, 1, 0),
            Conv2d(16, 512, 1, 1, 0),
            Conv2d(16, 512, 1, 1, 0),
            Conv2d(16, 512, 1, 1, 0)
        ])

        self.face_encoder_layers = len(self.face_encoder)

        self.face_decoder_input_layer = Conv2d(512, 512, kernel_size=1, stride=1, padding=0)

        self.face_decode_layers = nn.ModuleList([
            nn.Sequential(
                Conv2dTranspose(1024, 512, 4, 1, 0, 0),
                Conv2d(512, 512, 3, 1, 1, normalize=False, residual=True),
            ),
            faceDecodingBlock(1024, 512, 3, 2, 1, output_padding=1), # 8, 8
            faceDecodingBlock(1024, 512, 3, 2, 1, output_padding=1), # 16, 16
            faceDecodingBlock(1024, 256, 3, 2, 1,output_padding=1), # 32, 32
            faceDecodingBlock(512, 128, 3, 2, 1,output_padding=1), # 64, 64
            faceDecodingBlock(256, 64, 3, 2, 1,output_padding=1) # 128, 128
        ])

        self.toRGB = nn.ModuleList([
            nn.Sequential(
                Conv2d(512, 32, 3, 1, 1),
                nn.Conv2d(32, 3, 1, 1, 0),
                nn.Sigmoid()
            ),
            nn.Sequential(
                Conv2d(512, 32, 3, 1, 1),
                nn.Conv2d(32, 3, 1, 1, 0),
                nn.Sigmoid()
            ),
            nn.Sequential(
                Conv2d(512, 32, 3, 1, 1),
                nn.Conv2d(32, 3, 1, 1, 0),
                nn.Sigmoid()
            ),
            nn.Sequential(
                Conv2d(256, 32, 3, 1, 1),
                nn.Conv2d(32, 3, 1, 1, 0),
                nn.Sigmoid()
            ),
            nn.Sequential(
                Conv2d(128, 32, 3, 1, 1),
                nn.Conv2d(32, 3, 1, 1, 0),
                nn.Sigmoid()
            ),
            nn.Sequential(
                Conv2d(64, 32, 3, 1, 1),
                nn.Conv2d(32, 3, 1, 1, 0),
                nn.Sigmoid()
            ),
        ])

    def forward(self, audio_sequences, face_sequences, step=0):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1

        feats = []
        x = face_sequences
        x = self.from6to16(x)
        
        for i in range(step, -1, -1):
            index = self.face_encoder_layers - i - 1
            
            if i == step:
                x = self.from16[index](x)                
                
            x = self.face_encoder[index](x)
            feats.append(x)

        x = audio_embedding

        x = self.face_decoder_input_layer(x)
        
        for i in range(0, step + 1):     
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                    print(i)
                    print(x.size())
                    print(feats[-1].size())
                    raise e
            
            feats.pop()

            x = self.face_decode_layers[i](x)

        x = self.toRGB[step](x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x
        
        return outputs
