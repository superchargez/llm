import os
import sys
import logging
import win32com.client
from win32com.client import constants

# Setup logging to include detailed timestamps and levels.
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def validate_range(requested_start, requested_end, max_count, file_type="document"):
    """
    Validates the requested range against the max_count (pages for Word or slides for PPTX).
    If the requested_end exceeds max_count, the user is offered to extract the first half instead.
    Returns a tuple (start, end) if valid, or None if the user aborts.
    """
    if requested_start < 1 or requested_start > max_count:
        logging.error(f"Requested start {requested_start} is out of bounds. Valid range is 1 to {max_count}.")
        return None
    if requested_end > max_count:
        logging.error(f"Requested end {requested_end} is out of bounds. The document has only {max_count} pages/slides.")
        choice = input(f"The requested end ({requested_end}) exceeds the maximum ({max_count}). "
                        f"Would you like to extract the first half instead? (Y/N): ").strip().lower()
        if choice == 'y':
            new_end = max_count // 2
            logging.info(f"User opted to extract first half: range will be 1 to {new_end}.")
            return (1, new_end)
        else:
            logging.error("User chose not to adjust the range. Skipping this range.")
            return None
    return (requested_start, requested_end)

def split_word_document(input_file, ranges):
    logging.info("Word Document Splitter started")
    logging.info("Checking if Microsoft Word is accessible...")
    try:
        word_app = win32com.client.Dispatch("Word.Application")
        logging.info("Microsoft Word is accessible")
    except Exception as e:
        logging.error(f"Error accessing Microsoft Word: {e}")
        return

    word_app.Visible = False  # run in background

    try:
        doc_path = os.path.abspath(input_file)
        logging.info(f"Word started, opening document: {doc_path}")
        doc = word_app.Documents.Open(doc_path)
        logging.info("Document opened, calculating page count...")

        # Fallback for wdStatisticPages (normally 2)
        try:
            pages_const = constants.wdStatisticPages
        except AttributeError:
            pages_const = 2
            logging.warning("wdStatisticPages constant not found. Using fallback value 2.")

        try:
            total_pages = doc.ComputeStatistics(pages_const)
            logging.info(f"Page count determined: {total_pages} pages")
        except Exception as e:
            logging.error(f"Error getting page count: {e}")
            total_pages = None
            logging.warning("Unable to determine page count")

        for (req_start, req_end) in ranges:
            valid_range = validate_range(req_start, req_end, total_pages, file_type="Word document")
            if not valid_range:
                continue  # Skip invalid range
            start_page, end_page = valid_range

            logging.info(f"Processing range: pages {start_page} to {end_page}")
            for page in range(start_page, end_page + 1):
                logging.debug(f"Processing page {page}")

            # Fallback for wdGoToPage and wdGoToAbsolute constants (both normally 1)
            try:
                goto_page = constants.wdGoToPage
            except AttributeError:
                goto_page = 1
                logging.warning("wdGoToPage constant not found. Using fallback value 1.")
            try:
                goto_absolute = constants.wdGoToAbsolute
            except AttributeError:
                goto_absolute = 1
                logging.warning("wdGoToAbsolute constant not found. Using fallback value 1.")

            try:
                rng_start = doc.GoTo(What=goto_page, Which=goto_absolute, Count=start_page)
            except Exception as e:
                logging.error(f"Error using GoTo for start page {start_page}: {e}")
                continue

            if end_page < total_pages:
                try:
                    rng_end = doc.GoTo(What=goto_page, Which=goto_absolute, Count=end_page + 1)
                    # Adjust to end at the very end of previous page
                    rng_end.Start = rng_end.Start - 1
                except Exception as e:
                    logging.error(f"Error using GoTo for end page {end_page}: {e}")
                    continue
            else:
                rng_end = doc.Content

            try:
                page_range = doc.Range(rng_start.Start, rng_end.Start)
                logging.info(f"Extracted content from page {start_page} to {end_page}")
            except Exception as e:
                logging.error(f"Error extracting range: {e}")
                continue

            # Create a new document and set its PageSetup to match the original document.
            new_doc = word_app.Documents.Add()
            new_doc.PageSetup.TopMargin = doc.PageSetup.TopMargin
            new_doc.PageSetup.BottomMargin = doc.PageSetup.BottomMargin
            new_doc.PageSetup.LeftMargin = doc.PageSetup.LeftMargin
            new_doc.PageSetup.RightMargin = doc.PageSetup.RightMargin
            new_doc.PageSetup.PageWidth = doc.PageSetup.PageWidth
            new_doc.PageSetup.PageHeight = doc.PageSetup.PageHeight
            new_doc.PageSetup.Orientation = doc.PageSetup.Orientation

            new_doc.Content.FormattedText = page_range.FormattedText

            # Remove any trailing empty paragraphs that might add extra pages.
            while new_doc.Paragraphs.Count > 0:
                last_para = new_doc.Paragraphs(new_doc.Paragraphs.Count)
                if last_para.Range.Text.strip() == "":
                    last_para.Range.Delete()
                else:
                    break

            base, _ = os.path.splitext(doc_path)
            output_file = os.path.join(os.path.dirname(doc_path),
                                       f"{os.path.basename(base)}_pages_{start_page}_{end_page}.docx")
            new_doc.SaveAs(os.path.abspath(output_file))
            logging.info(f"Saved split document as {output_file}")
            new_doc.Close(False)

        doc.Close(False)
    except Exception as e:
        logging.error(f"Error processing Word document: {e}")
    finally:
        word_app.Quit()
        logging.info("Word application closed")

def split_pptx_file(input_file, ranges):
    logging.info("PowerPoint Document Splitter started")
    logging.info("Checking if Microsoft PowerPoint is accessible...")
    try:
        ppt_app = win32com.client.Dispatch("PowerPoint.Application")
        logging.info("Microsoft PowerPoint is accessible")
    except Exception as e:
        logging.error(f"Error accessing Microsoft PowerPoint: {e}")
        return

    ppt_app.Visible = True

    try:
        pres_path = os.path.abspath(input_file)
        logging.info(f"PowerPoint started, opening presentation: {pres_path}")
        presentation = ppt_app.Presentations.Open(pres_path, WithWindow=False)
        total_slides = presentation.Slides.Count
        logging.info(f"Presentation opened, total slides determined: {total_slides}")

        for (req_start, req_end) in ranges:
            valid_range = validate_range(req_start, req_end, total_slides, file_type="PPTX presentation")
            if not valid_range:
                continue
            start_slide, end_slide = valid_range

            logging.info(f"Processing slide range: {start_slide} to {end_slide}")
            for slide_num in range(start_slide, end_slide + 1):
                logging.debug(f"Processing slide {slide_num}")

            new_pres = ppt_app.Presentations.Add()

            for i in range(start_slide, end_slide + 1):
                if i > total_slides:
                    logging.warning(f"Slide {i} does not exist in the original presentation")
                    continue
                try:
                    slide = presentation.Slides(i)
                    slide.Copy()
                    new_slide = new_pres.Slides.Paste()
                    logging.info(f"Copied slide {i}")
                except Exception as e:
                    logging.error(f"Error copying slide {i}: {e}")

            base, _ = os.path.splitext(pres_path)
            output_file = os.path.join(os.path.dirname(pres_path),
                                       f"{os.path.basename(base)}_slides_{start_slide}_{end_slide}.pptx")
            new_pres.SaveAs(os.path.abspath(output_file))
            logging.info(f"Saved split presentation as {output_file}")
            new_pres.Close()
        presentation.Close()
    except Exception as e:
        logging.error(f"Error processing PPTX: {e}")
    finally:
        ppt_app.Quit()
        logging.info("PowerPoint application closed")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python splitter.py <input_file> <range1> [<range2> ...]")
        print("Example: python splitter.py document.docx 1-3 5-7")
        sys.exit(1)

    input_file = sys.argv[1]
    logging.info(f"User entered input file: {input_file}")
    output_dir = os.getcwd()
    logging.info(f"User entered output directory: {output_dir}")
    
    range_args = sys.argv[2:]
    ranges = []
    for r in range_args:
        try:
            parts = r.split('-')
            if len(parts) != 2:
                raise ValueError("Invalid range format")
            start = int(parts[0])
            end = int(parts[1])
            ranges.append((start, end))
        except Exception as e:
            logging.error(f"Invalid range '{r}': {e}")
            sys.exit(1)

    ext = os.path.splitext(input_file)[1].lower()
    if ext in [".doc", ".docx"]:
        split_word_document(input_file, ranges)
    elif ext in [".ppt", ".pptx"]:
        split_pptx_file(input_file, ranges)
    else:
        logging.error("Unsupported file type. Only .doc/.docx and .ppt/.pptx files are supported.")
