from rfdetr import RFDETRNano
import sys
import os

def main(argv):  

    VERSION = 1
    EPOCHS = 100

    for i in range(len(argv)):
        if argv[i] == "--version":
            VERSION = int(argv[i+1])
        if argv[i] == "--epochs":
            EPOCHS = int(argv[i+1])

    subversion = 0
    while os.path.isdir(f"./models/rfdetr_nano_v{VERSION}.{subversion}"):
        subversion += 1

    model = RFDETRNano(pretrain_weights="rf-detr-nano.pth")

    print(f"Starting training v{VERSION}.{subversion} for {EPOCHS} epochs...")

    model.train(
        dataset_dir=f"./datasets/v_{VERSION}",
        weight_decay=0.0005,
        epochs=EPOCHS,
        batch_size=8,
        grad_accum_steps=2,
        use_ema=True,
        lr=1e-4,
        #lr_encoder=1e-5,
        output_dir=f"./models/rfdetr_nano_v{VERSION}.{subversion}",
        early_stopping=True,
        tensorboard=True,
        early_stopping_patience=15,
    )

if __name__ == "__main__":
    main(sys.argv)