import qrcode  # Import library for QR code generation
from PIL import Image  # Import library for image manipulation

def main():  # Define the main function
    """
    Generates a QR code that embeds a link to an image.
    """

    # Create a QRCode object with specified parameters:
    qr = qrcode.QRCode(
        version=2,               # QR code version (adjust for data capacity)
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=10,             # Size of individual QR code boxes
        border=4                 # Border width around the QR code
    )

    # Add the image URL as data to the QR code:
    qr.add_data("https://i.ibb.co/6Fgtyw0/IMG-7742.jpg")

    # Make the QR code with automatic fitting:
    qr.make(fit=True)

    # Generate the QR code image with black and white colors:
    img = qr.make_image(fill_color="black", back_color="white")

    # Save the QR code image as a PNG file:
    img.save("qr_code_with_embedded_image.png")

# Execute the main function if the script is run directly:
if __name__ == '__main__':
    main()
