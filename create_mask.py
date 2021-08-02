from utils import folder_operations as fo
from utils import superpixel, matrix

def main():
    isUpperGland = True
    create_mask_matrix = True
    create_mask_image = True

    if isUpperGland:
        clahe_images, filenames = fo.read_CLAHE_Up()
    else:
        clahe_images, filenames = fo.read_CLAHE_Low()

    n_segments = 180
    if create_mask_matrix:
        for i in range(0, len(clahe_images)):
            print(i, filenames[i])
            segmentedImage, segments = superpixel.createSuperpixelWithMask(clahe_images[i], n_segments)
            mask = superpixel.selectPoint(segmentedImage, segments)
            if isUpperGland:
                matrix.create_matrix_mask(mask, "Masks/Lower Gland/mask_" + filenames[i])
            else:
                matrix.create_matrix_mask(mask, "Masks/Upper Gland/mask_" + filenames[i])

    if create_mask_image:
        if isUpperGland:
            clahe_images, masks, filenames = fo.read_Masks_CLAHE_Up()
            fo.create_masks_images_Up(masks, filenames)
        else:
            clahe_images, masks, filenames = fo.read_Masks_CLAHE_Low()
            fo.create_masks_images_Low(masks, filenames)


if __name__ == "__main__":
    main()
