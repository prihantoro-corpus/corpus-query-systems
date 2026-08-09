import streamlit.components.v1 as components

def render_print_button():
    """
    Renders a button that triggers the browser's print dialog,
    defaulting to landscape mode via CSS injection.
    """
    components.html(
        """
        <style>
        @media print {
           @page { size: landscape; }
           /* Hide the print button itself during printing */
           .print-btn-container { display: none !important; }
        }
        body { margin: 0; font-family: "Source Sans Pro", sans-serif; }
        .print-btn {
            background-color: #f0f2f6;
            color: #31333F;
            border: 1px solid #d5d9e0;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 400;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s, border-color 0.2s, color 0.2s;
        }
        .print-btn:hover {
            border-color: #ff4b4b;
            color: #ff4b4b;
        }
        </style>
        <div class="print-btn-container">
            <button class="print-btn" onclick="window.parent.print()">🖨️ Print Report to PDF</button>
        </div>
        """,
        height=50
    )
