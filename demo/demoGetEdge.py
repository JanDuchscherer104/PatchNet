import cv2
import numpy as np
import cv2
import numpy as np

def display_images(image, edges, contours, name=None):
    # Create a blank canvas to display the images side by side
    canvas = np.zeros((image.shape[0], image.shape[1] * 3, 3), dtype=np.uint8)

    # Draw the original image on the canvas
    canvas[:, :image.shape[1]] = image

    # Draw the edges on the canvas
    canvas[:, image.shape[1]:image.shape[1]*2] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Draw the contours on the canvas
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    canvas[:, image.shape[1]*2:] = image

    # Display the canvas
    cv2.imshow('Original, Edges, Contours', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save the image
    if name:
        cv2.imwrite(name, canvas)


def process_image(image_path, threshold_value, canny_threshold1, canny_threshold2, name=None):
    # Load the puzzle image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to turn non-white pixels into black
    _, threshold = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    print(threshold)

    # Apply Canny edge detection
    edges = cv2.Canny(threshold, canny_threshold1, canny_threshold2)
    
    # Find contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Call the new method
    display_images(image, edges, contours, name=name)


# Process the first image
# Here problem because background is white and some of the piece is white
process_image('demo/jigsaw-pieces-samples/onePiece.jpg', 240, 100, 200, 'jigsaw-pieces-samples/onePieceProcessed.jpg')

# Process the second image
# Here problem because background is non white but not consistent enough to be removed
process_image('demo/jigsaw-pieces-samples/realPiece.jpg', 100, 50, 150, 'jigsaw-pieces-samples/realPieceProcessed.jpg')

# Process the third image
# This image works fairly well because the background is consistent white and the piece's colors are distinct
process_image('demo/jigsaw-pieces-samples/VanGoghPiece.jpg', 240, 100, 200, 'jigsaw-pieces-samples/VanGoghPieceProcessed.jpg')
