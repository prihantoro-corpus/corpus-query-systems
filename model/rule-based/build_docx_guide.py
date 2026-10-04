import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_cell_background(cell, fill_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), fill_color)
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for margin_name, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_document():
    doc = docx.Document()
    
    # Page setup - Standard Margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        
    # Styles
    # Title
    p_title = doc.add_paragraph()
    run_title = p_title.add_run("CORTEX Rule-Based Tagger\nWord-Formation Operators Guide")
    run_title.font.name = 'Segoe UI'
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(30, 58, 138) # Dark Navy
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_after = Pt(18)

    def add_h1(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.name = 'Segoe UI'
        r.font.size = Pt(15)
        r.font.bold = True
        r.font.color.rgb = RGBColor(30, 58, 138)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.name = 'Segoe UI'
        r.font.size = Pt(12)
        r.font.bold = True
        r.font.color.rgb = RGBColor(51, 65, 85)
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        return p

    def add_p(text):
        p = doc.add_paragraph()
        r = p.add_run(text)
        r.font.name = 'Segoe UI'
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(51, 51, 51)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = 'Segoe UI'
            rb.font.size = Pt(10.5)
            rb.font.bold = True
            rb.font.color.rgb = RGBColor(30, 58, 138)
        r = p.add_run(text)
        r.font.name = 'Segoe UI'
        r.font.size = Pt(10.5)
        r.font.color.rgb = RGBColor(51, 51, 51)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        return p

    # 1. Overview
    add_h1("1. Overview")
    add_p("The Word-formation.txt file synthesizes full word forms from lemmas defined in Lemma.txt. This engine uses an explicit, cursor-based state machine and entry buffers (<E1>, <E2>) to execute complex morphological processes across languages:")
    add_bullet(" Simple and right-end character deletions before appending endings.", "Suffixation: ")
    add_bullet(" Left-side modifications and prefix insertions.", "Prefixation: ")
    add_bullet(" Internal stem insertions and deletions.", "Infixation: ")
    add_bullet(" Full, partial, and morphologically conditioned copying.", "Reduplication: ")

    # 2. Core Principles
    add_h1("2. Core Principles")
    add_bullet("The cursor starts at the Rightmost end of the lemma string. <E1> and <E2> both hold an exact copy of the original lemma.", "Initial Cursor State: ")
    add_bullet("Operators process left-to-right sequentially in the rule string.", "Sequential Execution: ")
    add_bullet("Irregular forms (e.g., foot > feet, sing > sang) should be hardcoded in the lexicon dictionary, while systematic sound patterns (e.g., consonant doubling in running) are handled by custom lemma codes (e.g., V015).", "Irregulars & Classes: ")

    # 3. Operator Reference Table
    add_h1("3. Operator Reference Table")
    
    headers = ["Category", "Operator", "Params", "Description", "Rule Example", "Before", "After"]
    data = [
        ["Base Form", "(empty)", "None", "Identity rule. If operator field is empty, output is unchanged lemma.", "V00 VERB", "fly", "fly"],
        ["Cursor Anchors", "<R>", "None", "Resets cursor to the Rightmost end of string.", "<D1>ies<R>ing", "fly", "fliesing"],
        ["Cursor Anchors", "<L>", "None", "Resets cursor to the Leftmost start of string (index 0).", "<L>un", "do", "undo"],
        ["Cursor Steps", "<N*k*>", "k (int)", "Moves cursor Right by k characters.", "<L><N1>em", "tali", "temali"],
        ["Cursor Steps", "<NB*k*>", "k (int)", "Moves cursor Back/Left by k characters.", "<NB1>g", "run", "rugn"],
        ["Deletions", "<D*k*>", "k (int)", "Deletes k characters to the right of cursor.", "<D1>ies", "fly", "flies"],
        ["Deletions", "<DN*k*>", "k (int)", "Deletes k characters immediately after inserted text.", "<L><N1>em<DN1>", "tali", "temli"],
        ["Reduplication", "<E1>", "Optional edits", "Represents Entry 1 (First copy of lemma stem).", "<E1>-<E2>", "orang", "orang-orang"],
        ["Reduplication", "<E2>", "Optional edits", "Represents Entry 2 (Second copy of lemma stem).", "<E1><E2>", "orang", "orangorang"],
        ["Grouping", "[...]", "Expression", "Encloses operations applied specifically to an entry block.", "[<E1><D1>ies]-[<E2><L>un]", "fly", "flies-unfly"]
    ]

    table = doc.add_table(rows=len(data)+1, cols=7)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    col_widths = [Inches(0.9), Inches(0.8), Inches(0.7), Inches(1.8), Inches(1.2), Inches(0.5), Inches(0.8)]

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].width = col_widths[i]
        p = hdr_cells[i].paragraphs[0]
        r = p.add_run(title)
        r.font.name = 'Segoe UI'
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        set_cell_background(hdr_cells[i], '1E3A8A') # Navy
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=100, right=100)

    # Data Rows
    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx+1].cells
        bg_color = 'F8FAFC' if r_idx % 2 == 0 else 'FFFFFF'
        for c_idx, cell_value in enumerate(row_data):
            row_cells[c_idx].width = col_widths[c_idx]
            p = row_cells[c_idx].paragraphs[0]
            r = p.add_run(cell_value)
            r.font.name = 'Consolas' if c_idx in [1, 4] else 'Segoe UI'
            r.font.size = Pt(9)
            r.font.color.rgb = RGBColor(30, 41, 59)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=100, bottom=100, left=100, right=100)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 4. Detailed Operator Mechanics & Examples
    add_h1("4. Detailed Operator Mechanics & Examples")

    add_h2("4.0 Base / Identity Form (No Operator)")
    add_p("If no operator command is specified for a code class (or left empty), the engine produces the unmodified lemma as the word form:")
    add_bullet("Rule: V00 VERB  | Input: fly > fly", "Base Identity Rule: ")

    add_h2("4.1 Suffixation (Right-End Modifications)")
    add_p("Default cursor position is at the right end (<R>).")
    add_bullet("Rule: V014 VERB ing | Input: see > seeing", "Simple Suffixation: ")
    add_bullet("Rule: V013 VERB <D1>ies | Input: fly > fl > flies", "Suffixation with Deletion (<Dk>): ")

    add_h2("4.2 Prefixation (Left-End Modifications)")
    add_p("Use <L> to move the cursor to the beginning of the word.")
    add_bullet("Rule: V015 VERB <L>un | Input: do > undo", "Simple Prefixation: ")
    add_bullet("Rule: V013 VERB <D1>ies<L>un | Input: fly > flies > unflies", "Prefixation + Suffixation Combined: ")

    add_h2("4.3 Infixation (Internal Stem Operations)")
    add_p("Use <Nk> to step forward from the left, or <NBk> to step backward from the right.")
    add_bullet("Rule: V020 VERB <L><N1>em | Input: tali > t|ali > temali", "Simple Infixation (<Nk>): ")
    add_bullet("Rule: V021 VERB <L><N1>em<DN1> | Input: tali > tem|ali > temli", "Infixation with Deletion (<DNk>): ")
    add_bullet("Rule: V022 VERB <L><N1>em<N2>o | Input: tali > temali > temaloi", "Multi-Point Infixation: ")

    add_h2("4.4 Reduplication (<E1>, <E2>)")
    add_p("Reduplication creates two entries (<E1> and <E2>) from the lemma stem, which can be combined with or without hyphens.")
    add_bullet("Rule: N001 NOUN <E1>-<E2> | Input: orang > orang-orang", "Full Reduplication (Delineated): ")
    add_bullet("Rule: N001 NOUN <E1><E2> | Input: orang > orangorang", "Full Reduplication (Non-delineated): ")
    add_bullet("Rule: N002 NOUN <E1>-[<E2><D1>] | Input: orang > orang-oran", "Partial / Modified Reduplication: ")
    add_bullet("Rule: V010 VERB [<E1><D1>ies]-[<E2><L>un] | Input: fly > flies-unfly", "Complex Morpho-Reduplication: ")

    add_h2("4.5 Class-Based Consonant Doubling")
    add_p("Instead of complex automatic phonological detection, assign class codes in Lemma.txt for stems requiring consonant doubling:")
    add_p("Lemma.txt:\nV015 run\nV015 thin\nV016 swim")
    add_p("Word-formation.txt:\nV015 VERB ning  # run > running, thin > thinning\nV016 VERB ming  # swim > swimming")

    # 5. Sample Word-formation.txt Master File
    add_h1("5. Sample Word-formation.txt Master File")
    
    sample_text = (
        "# Code    POS     Operator Command                     # Example / Description\n"
        "V00     VERB                                         # fly > fly (Unmodified Base Form)\n"
        "V013    VERB    <D1>ies                              # fly > flies\n"
        "V013    VERB    <D1>ies<L>un                         # fly > unflies\n"
        "V014    VERB    ing                                  # see > seeing\n"
        "V015    VERB    ning                                 # run > running\n"
        "V020    VERB    <L><N1>em                            # tali > temali\n"
        "V021    VERB    <L><N1>em<DN1>                       # tali > temli\n"
        "N001    NOUN    <E1>-<E2>                            # orang > orang-orang\n"
        "N002    NOUN    <E1>-[<E2><D1>]                      # orang > orang-oran\n"
        "V030    VERB    [<E1><D1>ies]-[<E2><L>un]            # fly > flies-unfly"
    )
    
    p_code = doc.add_paragraph()
    r_code = p_code.add_run(sample_text)
    r_code.font.name = 'Consolas'
    r_code.font.size = Pt(9.5)
    r_code.font.color.rgb = RGBColor(30, 41, 59)
    p_code.paragraph_format.space_before = Pt(6)
    p_code.paragraph_format.space_after = Pt(12)

    try:
        doc.save(r"C:\Users\priha\Documents\cortex\model\rule-based\word-formation-guide.docx")
        print("Successfully updated word-formation-guide.docx")
    except PermissionError:
        doc.save(r"C:\Users\priha\Documents\cortex\model\rule-based\word-formation-guide-v2.docx")
        print("Successfully saved to word-formation-guide-v2.docx (word-formation-guide.docx is locked/open in Word)")

if __name__ == "__main__":
    create_document()
