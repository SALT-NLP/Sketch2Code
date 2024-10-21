import os
from PIL import Image
import io
from playwright.sync_api import sync_playwright

def take_screenshot(url):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    screenshot_image = None

    try:
        with sync_playwright() as p:
            # Choose a browser, e.g., Chromium, Firefox, or WebKit
            browser = p.chromium.launch()
            page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=60000)

            # Take the screenshot and save it to a bytes buffer instead of a file
            image_bytes = page.screenshot(full_page=True, animations="disabled", timeout=60000)

            # Convert bytes to a PIL Image
            image_buffer = io.BytesIO(image_bytes)
            screenshot_image = Image.open(image_buffer)

            browser.close()
    except Exception as e: 
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image and return it
        screenshot_image = Image.new('RGB', (1280, 960), color='white')

    return screenshot_image


def take_and_save_screenshot(url, output_file="screenshot.png", do_it_again=False):
    # Convert local path to file:// URL if it's a file
    if os.path.exists(url):
        url = "file://" + os.path.abspath(url)

    # whether to overwrite existing screenshots
    if os.path.exists(output_file) and not do_it_again:
        print(f"{output_file} exists!")
        return

    try:
        with sync_playwright() as p:
            # Choose a browser, e.g., Chromium, Firefox, or WebKit
            browser = p.chromium.launch()
            page = browser.new_page()

            # Navigate to the URL
            page.goto(url, timeout=60000)

            # Take the screenshot
            page.screenshot(path=output_file, full_page=True, animations="disabled", timeout=60000)

            browser.close()
    except Exception as e: 
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image 
        img = Image.new('RGB', (1280, 960), color = 'white')
        img.save(output_file)


def take_screenshot_from_html(html_content):
    screenshot_image = None

    try:
        with sync_playwright() as p:
            # Launch the browser
            browser = p.chromium.launch()
            page = browser.new_page()

            # Set the HTML content
            page.set_content(html_content, timeout=60000)

            # Take the screenshot and save it to a bytes buffer instead of a file
            image_bytes = page.screenshot(full_page=True, animations="disabled", timeout=60000)

            # Convert bytes to a PIL Image
            image_buffer = io.BytesIO(image_bytes)
            screenshot_image = Image.open(image_buffer)

            browser.close()
    except Exception as e:
        print(f"Failed to take screenshot due to: {e}. Generating a blank image.")
        # Generate a blank image and return it
        screenshot_image = Image.new('RGB', (1280, 960), color='white')

    return screenshot_image