def st_main(args):
    import matplotlib.pyplot as plt
    from PIL import Image
    import re
    import torchvision

    from vgg16common import LOGGER
    from vgg16st.style_transfer import transfer_style

    # Regex pattern for matching the user input
    DEPTH_PATTERN = re.compile(r"([\+\-]?)(\d*)([\+\-]?)")

    # Set up the transforms
    LOGGER.info("Setting up transforms.")
    TO_TENSOR = torchvision.transforms.ToTensor()

    # Determine the depths of the model
    LOGGER.info("Calculating depths.")
    depths = args.depth
    model_depths = []

    for d in depths:
        groups = DEPTH_PATTERN.search(d).groups()
        if groups[0] == "-":
            model_depths.extend(range(1, int(groups[1]) + 1))
        elif groups[0] == "+":
            model_depths.extend(range(5, int(groups[1]) - 1, -1))
        elif groups[0] == "":
            if groups[2] == "-":
                model_depths.extend(range(int(groups[1]), 0, -1))
            elif groups[2] == "+":
                model_depths.extend(range(int(groups[1]), 6))
            else:
                model_depths.append(int(groups[1]))

    LOGGER.info(f"Depths to use: {model_depths}.")

    content_img = TO_TENSOR(Image.open(args.content).convert("RGB"))
    style_img = TO_TENSOR(Image.open(args.style).convert("RGB"))
    content = transfer_style(
        content_img,
        style_img,
        model_depths,
        use_gpu=args.gpu,
        alpha=args.alpha,
        method=args.method,
        square_size=args.square_size,
        model=args.model,
    )

    # Image plotting
    LOGGER.info("Generating the images.")
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(content_img.permute(1, 2, 0))
    ax[1].imshow(content.permute(1, 2, 0))
    ax[2].imshow(style_img.permute(1, 2, 0))

    ax[0].set_title("Original image")
    ax[1].set_title("Stylized image")
    ax[2].set_title("Style image")

    if args.save:
        LOGGER.info("Saving the transferred image.")
        fig.savefig(args.save, bbox_inches="tight")

    if args.plot:
        LOGGER.info("Showing the transferred image.")
        plt.show()
