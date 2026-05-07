from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import pandas as pd
from pathlib import Path

# ── Load data ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / 'train.csv', dayfirst=True,
                 parse_dates=['Order Date', 'Ship Date'])
df['shipping_days'] = (df['Ship Date'] - df['Order Date']).dt.days

synthetic_v1_path = BASE_DIR / 'synthetic_balanced_v1.csv'
synthetic_v2_path = BASE_DIR / 'synthetic_balanced_v2.csv'
v1 = pd.read_csv(synthetic_v1_path) if synthetic_v1_path.exists() else pd.DataFrame()
v2 = pd.read_csv(synthetic_v2_path) if synthetic_v2_path.exists() else pd.DataFrame()

# ── Document Setup ────────────────────────────────────────────────────────────
OUTPUT = str(BASE_DIR / 'RetailGPT_RAG_Knowledge_Base.pdf')
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2.2*cm, bottomMargin=2.2*cm
)

styles = getSampleStyleSheet()

# Custom styles
BRAND   = colors.HexColor('#1a3c5e')
ACCENT  = colors.HexColor('#e8a020')
LIGHT   = colors.HexColor('#f0f4f8')
MUTED   = colors.HexColor('#6b7a8d')

H1 = ParagraphStyle('H1', parent=styles['Title'],
    fontSize=22, textColor=BRAND, spaceAfter=10, leading=28,
    fontName='Helvetica-Bold')
H2 = ParagraphStyle('H2', parent=styles['Heading1'],
    fontSize=15, textColor=BRAND, spaceBefore=18, spaceAfter=6,
    fontName='Helvetica-Bold', borderPad=4)
H3 = ParagraphStyle('H3', parent=styles['Heading2'],
    fontSize=12, textColor=ACCENT, spaceBefore=12, spaceAfter=4,
    fontName='Helvetica-Bold')
BODY = ParagraphStyle('BODY', parent=styles['Normal'],
    fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
BULLET = ParagraphStyle('BULLET', parent=styles['Normal'],
    fontSize=10, leading=14, spaceAfter=3,
    leftIndent=16, bulletIndent=4)
QA = ParagraphStyle('QA', parent=styles['Normal'],
    fontSize=10, leading=14, spaceAfter=4,
    leftIndent=12, textColor=colors.HexColor('#1a1a2e'))
Q_STYLE = ParagraphStyle('Q', parent=styles['Normal'],
    fontSize=10, leading=14, spaceAfter=2,
    fontName='Helvetica-Bold', textColor=BRAND)
CAPTION = ParagraphStyle('CAPTION', parent=styles['Normal'],
    fontSize=8, textColor=MUTED, spaceAfter=4, alignment=TA_CENTER)
COVER_SUB = ParagraphStyle('CSUB', parent=styles['Normal'],
    fontSize=13, textColor=colors.white, alignment=TA_CENTER, leading=20)

def hr(): return HRFlowable(width='100%', thickness=1.2, color=ACCENT, spaceAfter=8, spaceBefore=4)
def thin_hr(): return HRFlowable(width='100%', thickness=0.4, color=MUTED, spaceAfter=6, spaceBefore=2)
def sp(h=6): return Spacer(1, h)

def tbl(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BRAND),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), 9),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('ALIGN',      (0,1), (0,-1), 'LEFT'),
        ('FONTSIZE',   (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT]),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#c0ccd8')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING',    (0,0), (-1,-1), 5),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
    ])
    t.setStyle(ts)
    return t

def qa(q, a):
    return [Paragraph(f'Q: {q}', Q_STYLE), Paragraph(f'A: {a}', QA), sp(4)]

def bullet(text): return Paragraph(f'• {text}', BULLET)

# ── Aggregate stats ───────────────────────────────────────────────────────────
total_sales   = df['Sales'].sum()
total_orders  = df['Order ID'].nunique()
total_cust    = df['Customer Name'].nunique()
total_prod    = df['Product Name'].nunique()
avg_order     = df.groupby('Order ID')['Sales'].sum().mean()

cat_sales = df.groupby('Category')['Sales'].agg(['sum','mean','count']).reset_index()
subcat_sales = df.groupby('Sub-Category')['Sales'].agg(['sum','mean','count']).reset_index().sort_values('sum', ascending=False)
region_sales = df.groupby('Region')['Sales'].agg(['sum','mean','count']).reset_index()
seg_sales    = df.groupby('Segment')['Sales'].agg(['sum','mean','count']).reset_index()
ship_sales   = df.groupby('Ship Mode')['Sales'].agg(['sum','mean','count']).reset_index()
ship_days    = df.groupby('Ship Mode')['shipping_days'].mean().reset_index()
state_sales  = df.groupby('State')['Sales'].sum().sort_values(ascending=False).head(15).reset_index()
city_sales   = df.groupby('City')['Sales'].sum().sort_values(ascending=False).head(15).reset_index()
top_cust     = df.groupby('Customer Name')['Sales'].sum().sort_values(ascending=False).head(20).reset_index()
top_prod     = df['Product Name'].value_counts().head(20).reset_index()
top_prod.columns = ['Product Name', 'Order Count']

# ── Story ─────────────────────────────────────────────────────────────────────
story = []

# ═══════════════════════════════════════════════════════════════════
# COVER PAGE
# ═══════════════════════════════════════════════════════════════════
from reportlab.platypus import FrameBreak
from reportlab.lib.units import inch

story.append(Spacer(1, 60))
story.append(Paragraph('RetailGPT', ParagraphStyle('cov1', fontSize=42,
    textColor=BRAND, alignment=TA_CENTER, fontName='Helvetica-Bold')))
story.append(Spacer(1, 8))
story.append(Paragraph('Complete RAG Knowledge Base', ParagraphStyle('cov2', fontSize=20,
    textColor=ACCENT, alignment=TA_CENTER, fontName='Helvetica-Bold')))
story.append(Spacer(1, 6))
story.append(Paragraph('Superstore Retail Intelligence Document', ParagraphStyle('cov3',
    fontSize=13, textColor=MUTED, alignment=TA_CENTER)))
story.append(Spacer(1, 30))
story.append(HRFlowable(width='60%', thickness=2, color=ACCENT, hAlign='CENTER'))
story.append(Spacer(1, 30))

cover_stats = [
    ['Metric', 'Value'],
    ['Total Transactions', f'{len(df):,}'],
    ['Total Revenue', f'${total_sales:,.2f}'],
    ['Unique Orders', f'{total_orders:,}'],
    ['Unique Customers', f'{total_cust:,}'],
    ['Unique Products', f'{total_prod:,}'],
    ['Average Order Value', f'${avg_order:,.2f}'],
    ['Product Categories', '3'],
    ['Sub-Categories', '17'],
    ['US States Covered', f"{df['State'].nunique()}"],
    ['Cities Served', f"{df['City'].nunique()}"],
]
story.append(tbl(cover_stats, col_widths=[9*cm, 6*cm]))
story.append(Spacer(1, 30))
story.append(Paragraph('Version 1.0  |  RetailGPT Project  |  Superstore Dataset',
    ParagraphStyle('footer', fontSize=9, textColor=MUTED, alignment=TA_CENTER)))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('Table of Contents', H1))
story.append(hr())
toc_items = [
    ('1', 'Business Overview & Dataset Description', '3'),
    ('2', 'Product Catalog — Categories & Sub-Categories', '4'),
    ('3', 'Sales Performance Analytics', '6'),
    ('4', 'Regional & Geographic Intelligence', '8'),
    ('5', 'Customer Segments & Profiles', '10'),
    ('6', 'Shipping & Logistics', '12'),
    ('7', 'Customer Intelligence — Top Buyers & Frequency', '14'),
    ('8', 'Product Intelligence — Best Sellers & High Value', '15'),
    ('9', 'Data Versions & Synthetic Datasets', '17'),
    ('10', 'System Architecture — RetailGPT Azure Deployment', '18'),
    ('11', 'FAQ — Frequently Asked Questions (100+)', '20'),
    ('12', 'Glossary of Terms', '30'),
]
for num, title, page in toc_items:
    story.append(Paragraph(
        f'<b>{num}.</b>&nbsp;&nbsp;{title}&nbsp;&nbsp;<font color="#6b7a8d">................... {page}</font>',
        ParagraphStyle('toc', fontSize=11, leading=20, leftIndent=10)))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — BUSINESS OVERVIEW
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('1. Business Overview & Dataset Description', H1))
story.append(hr())

story.append(Paragraph('About the Superstore Dataset', H2))
story.append(Paragraph(
    'The RetailGPT system is built on the classic Superstore retail dataset — a widely used, '
    'industry-standard dataset representing a fictional US-based retail company selling products '
    'across three major categories: Furniture, Office Supplies, and Technology. The dataset covers '
    'multi-year transactional sales data across all four US geographic regions and 49 states, '
    'making it ideal for retail analytics, AI-powered Q&A, and business intelligence workloads.', BODY))

story.append(Paragraph('Dataset Snapshot', H3))
snap = [
    ['Field', 'Details'],
    ['Dataset Name', 'Superstore Sales — Retail Transactions'],
    ['Total Rows (train.csv)', '9,800 transactions'],
    ['Columns', '18 fields per row'],
    ['Date Range', '2015 – 2018 (multi-year)'],
    ['Geography', 'United States — 49 states, 531 cities'],
    ['Categories', 'Furniture, Office Supplies, Technology'],
    ['Sub-Categories', '17 product sub-categories'],
    ['Customer Segments', 'Consumer, Corporate, Home Office'],
    ['Shipping Modes', 'Standard Class, Second Class, First Class, Same Day'],
    ['Total Revenue', f'${total_sales:,.2f}'],
    ['Avg. Sale Value', f'${df["Sales"].mean():,.2f}'],
    ['Min. Sale Value', f'${df["Sales"].min():,.2f}'],
    ['Max. Sale Value', f'${df["Sales"].max():,.2f}'],
]
story.append(tbl(snap, col_widths=[8*cm, 9*cm]))
story.append(sp(8))

story.append(Paragraph('Column Definitions', H3))
cols_info = [
    ['Column', 'Type', 'Description'],
    ['Row ID', 'Integer', 'Unique sequential row identifier'],
    ['Order ID', 'String', 'Unique order identifier (e.g., CA-2017-152156)'],
    ['Order Date', 'Date', 'Date the order was placed (DD/MM/YYYY)'],
    ['Ship Date', 'Date', 'Date the order was shipped'],
    ['Ship Mode', 'Categorical', 'Shipping speed: Standard Class, Second Class, First Class, Same Day'],
    ['Customer ID', 'String', 'Unique customer identifier (e.g., CG-12520)'],
    ['Customer Name', 'String', 'Full name of the customer'],
    ['Segment', 'Categorical', 'Customer segment: Consumer, Corporate, Home Office'],
    ['Country', 'String', 'Country of sale (always United States)'],
    ['City', 'String', 'City where order was shipped'],
    ['State', 'String', 'US state of delivery'],
    ['Postal Code', 'String', 'Zip code of delivery address'],
    ['Region', 'Categorical', 'US region: East, West, Central, South'],
    ['Product ID', 'String', 'Unique product SKU identifier'],
    ['Category', 'Categorical', 'Main product category'],
    ['Sub-Category', 'Categorical', 'Product sub-category (17 types)'],
    ['Product Name', 'String', 'Full name of the product'],
    ['Sales', 'Float', 'Revenue generated by this line item (USD)'],
]
story.append(tbl(cols_info, col_widths=[4*cm, 3*cm, 10*cm]))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — PRODUCT CATALOG
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('2. Product Catalog — Categories & Sub-Categories', H1))
story.append(hr())

story.append(Paragraph(
    'RetailGPT covers 1,849 unique products organized into 3 main categories and 17 sub-categories. '
    'Products range from basic office stationery to high-value technology equipment and premium furniture.',
    BODY))

# Category details
for cat, desc, subcats in [
    ('Furniture', 
     'The Furniture category encompasses all office and commercial furniture sold by the Superstore. '
     'It is the second-highest revenue category with $728,658 in total sales across 2,078 transactions. '
     'Average sale value is $350.65 — reflecting the higher price points of furniture items.',
     [
         ('Chairs', '607 orders | $322,822 total | $531.83 avg | 88 unique products',
          'Office and task chairs including executive leather chairs, ergonomic task chairs, '
          'folding chairs, and conference room chairs. High-value sub-category.'),
         ('Tables', '314 orders | $202,810 total | $645.89 avg | 56 unique products',
          'Conference tables, adjustable-height desks, rectangular tables, and meeting room '
          'tables. Highest average sale value among Furniture sub-categories.'),
         ('Bookcases', '226 orders | $113,813 total | $503.60 avg | 50 unique products',
          'Wooden and laminate bookcases for office use; includes wall units and freestanding shelves.'),
         ('Furnishings', '931 orders | $89,212 total | $95.82 avg | 186 unique products',
          'Accessories and small furniture items: lamps, picture frames, clocks, rugs, curtains, '
          'coat racks. Most frequently ordered Furniture sub-category.'),
     ]),
    ('Office Supplies',
     'Office Supplies is the highest-volume category with 5,909 transactions — accounting for 60% '
     'of all orders. Total revenue is $705,422 with an average sale of $119.38. This category '
     'contains everyday consumables and office essentials that drive repeat purchasing.',
     [
         ('Binders', '1,492 orders | $200,028 total | $134.07 avg | 211 unique products',
          'Ring binders, presentation binders, report covers. Includes brands like Avery, GBC, '
          'Cardinal, and Storex. Very high order frequency.'),
         ('Paper', '1,338 orders | $76,828 total | $57.42 avg | 277 unique products',
          'Largest product catalog with 277 SKUs. Includes copy paper, specialty paper, carbonless '
          'paper, photo paper. Low unit price but very high volume.'),
         ('Storage', '832 orders | $219,343 total | $263.63 avg | 131 unique products',
          'Filing cabinets, desk organizers, stackable drawers, hanging file folders, '
          'archival boxes, and personal safes.'),
         ('Art', '785 orders | $26,705 total | $34.02 avg | 157 unique products',
          'Pens, markers, pencils, colored pencils, correction fluid, dry-erase markers, '
          'highlighters, and art supplies.'),
         ('Appliances', '459 orders | $104,618 total | $227.93 avg | 97 unique products',
          'Electric kettles, coffee makers, fans, air purifiers, paper shredders, '
          'binding machines, and laminators.'),
         ('Labels', '357 orders | $12,347 total | $34.59 avg | 70 unique products',
          'Adhesive address labels, file folder labels, name badge labels, and label makers.'),
         ('Supplies', '184 orders | $46,420 total | $252.28 avg | 36 unique products',
          'Toner cartridges, ink cartridges, printer ribbons, and specialty office materials.'),
         ('Envelopes', '248 orders | $16,128 total | $65.03 avg | 44 unique products',
          'Business envelopes, padded mailers, bubble mailers, and security envelopes.'),
         ('Fasteners', '214 orders | $3,001 total | $14.03 avg | 34 unique products',
          'Rubber bands, binder clips, paper clips, push pins, and staple removers. '
          'Lowest average sale value overall.'),
     ]),
    ('Technology',
     'Technology is the highest-revenue category with $827,455 in total sales from 1,813 transactions. '
     'Average sale value is $456.40 — the highest of all three categories. Technology products '
     'include computing, communication, and electronic devices.',
     [
         ('Phones', '876 orders | $327,782 total | $374.18 avg | 189 unique products',
          'Smartphones, desk phones, cordless phones, and phone accessories. Most ordered '
          'Technology sub-category.'),
         ('Accessories', '756 orders | $164,186 total | $217.18 avg | 147 unique products',
          'Computer peripherals: mice, keyboards, headsets, webcams, USB hubs, monitor stands, '
          'and cables.'),
         ('Machines', '115 orders | $189,238 total | $1,645.55 avg | 63 unique products',
          'High-value items: fax machines, label printers, binding equipment. Second highest '
          'average sale value per unit.'),
         ('Copiers', '66 orders | $146,248 total | $2,215.88 avg | 13 unique products',
          'Photocopiers and all-in-one printers. Fewest orders but HIGHEST average sale value '
          'at $2,215.88. Single copier order = significant revenue.'),
     ]),
]:
    story.append(Paragraph(f'Category: {cat}', H2))
    story.append(Paragraph(desc, BODY))
    story.append(Paragraph('Sub-Categories:', H3))
    for scname, stats, sdesc in subcats:
        story.append(KeepTogether([
            Paragraph(f'<b>{scname}</b> — {stats}', ParagraphStyle('sc',
                fontSize=10, textColor=BRAND, fontName='Helvetica-Bold',
                spaceBefore=6, spaceAfter=2, leftIndent=10)),
            Paragraph(sdesc, ParagraphStyle('scd', fontSize=9, leading=13,
                textColor=colors.HexColor('#333333'), leftIndent=20, spaceAfter=4)),
        ]))
    story.append(sp(4))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — SALES PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('3. Sales Performance Analytics', H1))
story.append(hr())

story.append(Paragraph('Overall Sales Summary', H2))
summary = [
    ['Metric', 'Value'],
    ['Total Revenue (all transactions)', f'${total_sales:,.2f}'],
    ['Total Number of Transactions', f'{len(df):,}'],
    ['Unique Orders', f'{total_orders:,}'],
    ['Average Revenue per Transaction', f'${df["Sales"].mean():,.2f}'],
    ['Median Revenue per Transaction', f'${df["Sales"].median():,.2f}'],
    ['Minimum Transaction Value', f'${df["Sales"].min():,.4f}'],
    ['Maximum Transaction Value', f'${df["Sales"].max():,.2f}'],
    ['25th Percentile (Q1)', f'${df["Sales"].quantile(0.25):,.2f}'],
    ['75th Percentile (Q3)', f'${df["Sales"].quantile(0.75):,.2f}'],
    ['Standard Deviation of Sales', f'${df["Sales"].std():,.2f}'],
    ['Average Order Value (by Order ID)', f'${avg_order:,.2f}'],
]
story.append(tbl(summary, col_widths=[10*cm, 7*cm]))
story.append(sp(10))

story.append(Paragraph('Sales by Category', H2))
cat_tbl_data = [['Category', 'Total Sales ($)', 'Avg Sale ($)', 'Transactions', '% of Revenue']]
for _, row in cat_sales.iterrows():
    pct = row['sum'] / total_sales * 100
    cat_tbl_data.append([
        row['Category'], f"${row['sum']:,.2f}", f"${row['mean']:,.2f}",
        f"{int(row['count']):,}", f"{pct:.1f}%"
    ])
story.append(tbl(cat_tbl_data, col_widths=[5*cm, 4*cm, 4*cm, 3.5*cm, 3.5*cm]))
story.append(sp(6))
story.append(Paragraph(
    'Technology leads in revenue ($827,455) despite having the fewest transactions (1,813), '
    'reflecting high-value unit pricing. Office Supplies dominates by transaction count (5,909 = 60% '
    'of all orders) but has a low average sale of $119. Furniture sits in the middle — high average '
    'value ($350) but moderate volume.', BODY))

story.append(Paragraph('Sales by Sub-Category (All 17)', H2))
subcat_data = [['Sub-Category', 'Category', 'Total Sales ($)', 'Avg Sale ($)', 'Orders', 'Unique Products']]
prod_counts = df.groupby('Sub-Category')['Product Name'].nunique()
cat_map = df.groupby('Sub-Category')['Category'].first()
for _, row in subcat_sales.iterrows():
    subcat_data.append([
        row['Sub-Category'],
        cat_map[row['Sub-Category']],
        f"${row['sum']:,.2f}",
        f"${row['mean']:,.2f}",
        f"{int(row['count']):,}",
        f"{prod_counts[row['Sub-Category']]}",
    ])
story.append(tbl(subcat_data, col_widths=[3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm, 2.5*cm, 3.5*cm]))
story.append(sp(6))

story.append(Paragraph('Key Insights:', H3))
for insight in [
    'Phones ($327,782) is the top revenue sub-category — highest total sales overall.',
    'Chairs ($322,822) is a close second and the top Furniture sub-category.',
    'Copiers have the highest average sale value at $2,215.88 per transaction.',
    'Machines average $1,645.55 per order — premium items with low order frequency.',
    'Fasteners are the cheapest sub-category at $14.03 average — consumables.',
    'Paper has the most SKUs (277 products) but very low average sale ($57.42).',
    'Tables have the fewest orders in Furniture (314) but highest Furniture avg ($645.89).',
    'Binders and Storage are the highest-volume Office Supplies sub-categories.',
]:
    story.append(bullet(insight))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — REGIONAL & GEOGRAPHIC
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('4. Regional & Geographic Intelligence', H1))
story.append(hr())

story.append(Paragraph('Sales by US Region', H2))
reg_data = [['Region', 'Total Sales ($)', 'Avg Sale ($)', 'Orders', '% of Revenue']]
for _, row in region_sales.iterrows():
    pct = row['sum'] / total_sales * 100
    reg_data.append([
        row['Region'], f"${row['sum']:,.2f}", f"${row['mean']:,.2f}",
        f"{int(row['count']):,}", f"{pct:.1f}%"
    ])
story.append(tbl(reg_data, col_widths=[4*cm, 5*cm, 4*cm, 4*cm, 4*cm]))
story.append(sp(6))
story.append(Paragraph(
    'The West region leads in both total revenue ($710,219 = 31.6%) and transaction count (3,140 orders). '
    'The East is second at $669,518. Central and South trail significantly. Despite having the fewest '
    'transactions (1,598), the South achieves the highest average sale value ($243.52) among all regions.', BODY))

story.append(Paragraph('Top 15 States by Revenue', H2))
state_data = [['Rank', 'State', 'Total Sales ($)', 'Region']]
state_region = df.groupby('State')['Region'].first().to_dict()
for i, (_, row) in enumerate(state_sales.iterrows(), 1):
    state_data.append([
        str(i), row['State'], f"${row['Sales']:,.2f}",
        state_region.get(row['State'], 'N/A')
    ])
story.append(tbl(state_data, col_widths=[2.5*cm, 6*cm, 5*cm, 4.5*cm]))
story.append(sp(6))
story.append(Paragraph(
    'California dominates with $446,306 in revenue — nearly 1.5x more than second-place New York '
    '($306,361). The top 3 states (California, New York, Texas) account for approximately 41% of '
    'total revenue. The top 5 states together generate over $1.17M (52% of total revenue).', BODY))

story.append(Paragraph('Top 15 Cities by Revenue', H2))
city_data = [['Rank', 'City', 'State', 'Total Sales ($)']]
city_state = df.groupby('City')['State'].first().to_dict()
for i, (_, row) in enumerate(city_sales.iterrows(), 1):
    city_data.append([
        str(i), row['City'], city_state.get(row['City'], ''),
        f"${row['Sales']:,.2f}"
    ])
story.append(tbl(city_data, col_widths=[2.5*cm, 5.5*cm, 5*cm, 5*cm]))
story.append(sp(6))
story.append(Paragraph(
    'New York City leads all cities at $252,462. Los Angeles ($173,420) and Seattle ($116,106) '
    'are strong second and third. The top 5 cities — New York City, Los Angeles, Seattle, San Francisco, '
    'and Philadelphia — together account for approximately $759,871 (34%) of total revenue.', BODY))

story.append(Paragraph('Geographic Coverage Summary', H3))
for fact in [
    f"States covered: {df['State'].nunique()} US states",
    f"Cities served: {df['City'].nunique()} unique cities",
    'All orders are domestic United States shipments only.',
    'West Region states include California, Washington, Oregon, Nevada, Arizona, Utah, Colorado, and others.',
    'East Region includes New York, Pennsylvania, Virginia, Florida, North Carolina, and New England states.',
    'Central Region covers Texas, Illinois, Ohio, Michigan, Wisconsin, Minnesota, and Midwest states.',
    'South Region includes Georgia, Tennessee, Alabama, Mississippi, Louisiana, Arkansas, and others.',
]:
    story.append(bullet(fact))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — CUSTOMER SEGMENTS
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('5. Customer Segments & Profiles', H1))
story.append(hr())

story.append(Paragraph(
    'The Superstore serves three distinct customer segments, each with different purchasing behaviors, '
    'average order values, and product preferences.', BODY))

story.append(Paragraph('Segment Performance Overview', H2))
seg_data = [['Segment', 'Total Sales ($)', 'Avg Sale ($)', 'Transactions', '% Revenue', '% Orders']]
total_orders_count = len(df)
for _, row in seg_sales.iterrows():
    rev_pct = row['sum'] / total_sales * 100
    ord_pct = row['count'] / total_orders_count * 100
    seg_data.append([
        row['Segment'], f"${row['sum']:,.2f}", f"${row['mean']:,.2f}",
        f"{int(row['count']):,}", f"{rev_pct:.1f}%", f"{ord_pct:.1f}%"
    ])
story.append(tbl(seg_data, col_widths=[3.5*cm, 4*cm, 3.5*cm, 3*cm, 3*cm, 3*cm]))
story.append(sp(8))

# Segment deep dives
seg_top_cats = df.groupby(['Segment','Category'])['Sales'].sum().reset_index()
for seg, desc in [
    ('Consumer',
     'The Consumer segment represents individual buyers — typically employees or professionals purchasing '
     'for personal or home use. With 5,101 transactions (52% of all orders) and $1.148M in revenue (51% '
     'of total), Consumers are the dominant customer type. They tend to purchase across all categories '
     'with a slight preference for Office Supplies due to frequency of small purchases. Average sale '
     'of $225 reflects a mix of small consumables and occasional high-value electronics.'),
    ('Corporate',
     'The Corporate segment represents businesses purchasing in bulk or for organizational needs. '
     '2,953 transactions generate $688,494 in revenue (30.6% of total). Average sale of $233 is '
     'slightly higher than Consumer. Corporate clients often buy Technology and Furniture in larger '
     'quantities and may have procurement contracts. They are high-value retention targets.'),
    ('Home Office',
     'The Home Office segment serves remote workers and small business owners. With 1,746 transactions '
     'and $424,982 in revenue (18.9%), this is the smallest but highest average-value segment at $243 '
     'per transaction. Home Office buyers tend to invest in quality equipment, ergonomic furniture, '
     'and productivity tools. Growing segment aligned with remote work trends.'),
]:
    story.append(Paragraph(f'Segment: {seg}', H2))
    story.append(Paragraph(desc, BODY))
    seg_subset = seg_top_cats[seg_top_cats['Segment'] == seg].sort_values('Sales', ascending=False)
    cat_breakdown = [['Category', 'Revenue from This Segment']]
    for _, r in seg_subset.iterrows():
        cat_breakdown.append([r['Category'], f"${r['Sales']:,.2f}"])
    story.append(tbl(cat_breakdown, col_widths=[8*cm, 9*cm]))
    story.append(sp(6))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — SHIPPING & LOGISTICS
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('6. Shipping & Logistics', H1))
story.append(hr())

story.append(Paragraph(
    'The Superstore offers four shipping modes with different speed-cost trade-offs. Understanding '
    'shipping patterns is essential for fulfillment planning, customer expectations, and SLA management.',
    BODY))

story.append(Paragraph('Shipping Mode Summary', H2))
# Merge ship_sales with ship_days
ship_merged = ship_sales.copy()
ship_merged = ship_merged.merge(
    df.groupby('Ship Mode')['shipping_days'].mean().reset_index().rename(
        columns={'shipping_days': 'avg_days'}), on='Ship Mode')
ship_merged = ship_merged.merge(
    df.groupby('Ship Mode')['shipping_days'].min().reset_index().rename(
        columns={'shipping_days': 'min_days'}), on='Ship Mode')
ship_merged = ship_merged.merge(
    df.groupby('Ship Mode')['shipping_days'].max().reset_index().rename(
        columns={'shipping_days': 'max_days'}), on='Ship Mode')

ship_tbl = [['Ship Mode', 'Orders', 'Total Revenue', 'Avg Sale', 'Avg Days', 'Min Days', 'Max Days']]
for _, row in ship_merged.iterrows():
    ship_tbl.append([
        row['Ship Mode'], f"{int(row['count']):,}", f"${row['sum']:,.2f}",
        f"${row['mean']:,.2f}", f"{row['avg_days']:.1f}", f"{int(row['min_days'])}",
        f"{int(row['max_days'])}"
    ])
story.append(tbl(ship_tbl, col_widths=[3.8*cm, 2.5*cm, 3.5*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm]))
story.append(sp(8))

story.append(Paragraph('Shipping Mode Details', H3))
for mode, days_range, cost_level, typical_use, order_pct in [
    ('Standard Class', '3–7 business days (avg 5.0)', 'Lowest cost',
     'Default option for most non-urgent orders. Ground shipping. '
     'Most popular with 5,859 orders (59.8% of all shipments). '
     'Suitable for bulk office supplies and non-time-sensitive furniture.',
     '59.8%'),
    ('Second Class', '1–5 business days (avg 3.2)', 'Low-medium cost',
     'Faster than Standard Class. Used for moderate-priority shipments. '
     '1,902 orders (19.4% of shipments). Good balance of speed and cost '
     'for corporate buyers with reasonable lead times.',
     '19.4%'),
    ('First Class', '1–4 business days (avg 2.2)', 'Medium-high cost',
     'Expedited shipping for urgent orders. 1,501 orders (15.3% of shipments). '
     'Frequently used for Technology items and time-sensitive supplies. '
     'Typically arrives within 2 business days.',
     '15.3%'),
    ('Same Day', '0–1 business days (avg 0.04)', 'Highest cost',
     'Premium same-day delivery. Only 538 orders (5.5% of shipments). '
     'Used for emergency office supply replenishment. Extremely fast but '
     'costs significantly more. Available in select cities only.',
     '5.5%'),
]:
    story.append(KeepTogether([
        Paragraph(f'<b>{mode}</b> ({order_pct} of orders)', ParagraphStyle(
            'sh', fontSize=11, textColor=BRAND, fontName='Helvetica-Bold',
            spaceBefore=8, spaceAfter=2, leftIndent=8)),
        Paragraph(f'Delivery time: {days_range}', BULLET),
        Paragraph(f'Cost level: {cost_level}', BULLET),
        Paragraph(typical_use, ParagraphStyle('shd', fontSize=9, leading=13,
            leftIndent=20, spaceAfter=6)),
    ]))

story.append(Paragraph('Shipping Insights', H3))
for insight in [
    'Standard Class accounts for 59.8% of all orders — most customers prefer economy shipping.',
    'Same Day delivery is rare (5.5%) suggesting most orders are planned rather than emergency.',
    'Average shipping time across all modes is approximately 4.2 days.',
    'First Class and Same Day together represent only 20.8% of orders — premium shipping is a niche.',
    'All four shipping modes have similar average sale values ($228–$236) — price does not strongly '
    'correlate with shipping choice.',
    'Shipping time does not include processing/warehouse preparation time — only transit days.',
]:
    story.append(bullet(insight))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — CUSTOMER INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('7. Customer Intelligence — Top Buyers & Frequency', H1))
story.append(hr())

story.append(Paragraph(
    f'The dataset contains {total_cust} unique customers. Customer analysis reveals high-value '
    'accounts and repeat buyers that represent the backbone of Superstore revenue.', BODY))

story.append(Paragraph('Top 20 Customers by Total Revenue', H2))
cust_data = [['Rank', 'Customer Name', 'Total Revenue', 'Avg per Transaction', 'Orders']]
cust_orders = df.groupby('Customer Name')['Order ID'].nunique()
cust_trans  = df.groupby('Customer Name').size()
for i, (_, row) in enumerate(top_cust.iterrows(), 1):
    avg = row['Sales'] / cust_trans.get(row['Customer Name'], 1)
    cust_data.append([
        str(i), row['Customer Name'], f"${row['Sales']:,.2f}",
        f"${avg:,.2f}", f"{cust_orders.get(row['Customer Name'], 0)}"
    ])
story.append(tbl(cust_data, col_widths=[2*cm, 6*cm, 4*cm, 4.5*cm, 3.5*cm]))
story.append(sp(6))
story.append(Paragraph(
    'Sean Miller is the top customer at $25,043 — driven largely by a single high-value Technology '
    'order (CA-2015-145317, $22,638). Tamara Chand and Raymond Buch follow with $19,052 and $15,117 '
    'respectively. The top 20 customers together account for approximately $239,000 in revenue.', BODY))

story.append(Paragraph('Most Frequently Ordering Customers', H2))
freq_cust = df.groupby('Customer Name').size().sort_values(ascending=False).head(15).reset_index()
freq_cust.columns = ['Customer Name', 'Transaction Count']
freq_cust_seg = df.groupby('Customer Name')['Segment'].first()
freq_data = [['Customer Name', 'Transactions', 'Segment']]
for _, row in freq_cust.iterrows():
    freq_data.append([row['Customer Name'], str(row['Transaction Count']),
                      freq_cust_seg.get(row['Customer Name'], 'N/A')])
story.append(tbl(freq_data, col_widths=[8*cm, 5*cm, 5*cm]))
story.append(sp(6))
story.append(Paragraph(
    'William Brown, Matt Abelman, and Paul Prost each have 34–35 transactions — they are the most '
    'active repeat buyers. High transaction frequency indicates loyal customers with ongoing supply needs.', BODY))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — PRODUCT INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('8. Product Intelligence — Best Sellers & High Value', H1))
story.append(hr())

story.append(Paragraph('Most Frequently Ordered Products (Top 20)', H2))
freq_prod_data = [['Rank', 'Product Name', 'Orders', 'Sub-Category']]
prod_subcat = df.groupby('Product Name')['Sub-Category'].first()
for i, (_, row) in enumerate(top_prod.iterrows(), 1):
    freq_prod_data.append([
        str(i), row['Product Name'][:55], str(row['Order Count']),
        prod_subcat.get(row['Product Name'], 'N/A')
    ])
story.append(tbl(freq_prod_data, col_widths=[2*cm, 10.5*cm, 2.5*cm, 5*cm]))
story.append(sp(6))
story.append(Paragraph(
    'Staple envelope (47 orders), Staples (46), and Easy-staple paper (44) dominate order frequency. '
    'These are low-cost consumables that customers repurchase regularly. The most ordered products are '
    'predominantly Office Supplies — confirming the high-volume, low-value nature of that category.', BODY))

story.append(Paragraph('Highest Single-Transaction Revenue Products', H2))
top_val = df.nlargest(15, 'Sales')[['Order ID','Customer Name','Product Name','Sub-Category','Sales']].reset_index(drop=True)
val_data = [['Rank', 'Product', 'Customer', 'Revenue']]
for i, row in top_val.iterrows():
    val_data.append([
        str(i+1), row['Product Name'][:45], row['Customer Name'], f"${row['Sales']:,.2f}"
    ])
story.append(tbl(val_data, col_widths=[2*cm, 9*cm, 5*cm, 4*cm]))
story.append(sp(6))

story.append(Paragraph('Revenue by Product (Top 15 Products)', H2))
top_rev_prod = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(15).reset_index()
top_rev_prod.columns = ['Product Name', 'Total Revenue']
rev_prod_subcat = df.groupby('Product Name')['Sub-Category'].first()
rev_prod_data = [['Rank', 'Product Name', 'Sub-Category', 'Total Revenue']]
for i, (_, row) in enumerate(top_rev_prod.iterrows(), 1):
    rev_prod_data.append([
        str(i), row['Product Name'][:50], rev_prod_subcat.get(row['Product Name'], 'N/A'),
        f"${row['Total Revenue']:,.2f}"
    ])
story.append(tbl(rev_prod_data, col_widths=[2*cm, 9*cm, 4*cm, 5*cm]))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — DATA VERSIONS
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('9. Data Versions & Synthetic Datasets', H1))
story.append(hr())

story.append(Paragraph(
    'The RetailGPT system uses three data files: the original training dataset (train.csv) and two '
    'synthetically generated balanced versions used for model training and evaluation.', BODY))

for name, fname, rows, mean_sales, median_sales, desc in [
    ('train.csv', 'Original Training Data', '9,800', '$230.77', '$54.49',
     'The primary Superstore dataset. Contains real transactional data with natural class imbalance — '
     'Office Supplies dominates at 60% of orders. Sales distribution is right-skewed with median $54.49 '
     'and outliers reaching $22,638. Used as the ground truth for analytics and RAG retrieval.'),
    ('synthetic_balanced_v1.csv', 'Synthetic Balanced Version 1', '9,800', '$2,044.41', '$207.02',
     'Synthetically generated dataset with balanced sub-category representation. Significantly higher '
     'mean sales ($2,044) suggesting upsampling of high-value transactions. Used for training ML models '
     'that require balanced class distribution. Paper (1,441), Binders (1,111), and Phones (1,015) '
     'are the most represented sub-categories.'),
    ('synthetic_balanced_v2.csv', 'Synthetic Balanced Version 2', '9,800', '$2,770.81', '$485.68',
     'Second synthetic version with even higher average sales ($2,770) and median ($485.68). '
     'Represents a more aggressive upsampling of high-value orders. Higher standard deviation '
     'suggests greater variability. Used as a complementary training dataset for robustness testing.'),
]:
    story.append(Paragraph(f'{fname} ({name})', H2))
    story.append(Paragraph(desc, BODY))
    story.append(tbl([
        ['Property', 'Value'],
        ['File Name', name],
        ['Total Rows', rows],
        ['Mean Sale', mean_sales],
        ['Median Sale', median_sales],
        ['Columns', '18 (same schema as train.csv)'],
    ], col_widths=[7*cm, 10*cm]))
    story.append(sp(8))

story.append(Paragraph('Dataset Schema Consistency', H3))
story.append(Paragraph(
    'All three datasets share an identical 18-column schema. This consistency ensures that '
    'any ETL pipeline, ML model, or RAG ingestion process built for train.csv works without '
    'modification on the synthetic datasets. The only differences are in value distributions '
    'and potential synthetic Customer IDs and Order IDs in the balanced versions.', BODY))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 10 — SYSTEM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('10. System Architecture — RetailGPT Azure Deployment', H1))
story.append(hr())

story.append(Paragraph(
    'RetailGPT is designed for enterprise deployment on Microsoft Azure. The architecture '
    'separates concerns across Frontend, Backend API, Identity/Security, Data Engineering, '
    'and Multi-Agent AI layers.', BODY))

story.append(Paragraph('Component Mapping: Local → Azure', H2))
arch_data = [
    ['Local Component', 'Azure Native Service', 'Rationale'],
    ['FastAPI Backend', 'Azure App Service\n(Web App for Containers)',
     'Serverless scaling, VNet integration, Docker support'],
    ['Streamlit Frontend', 'Azure App Service',
     'Separate tier decouples UI scaling from API load'],
    ['SQLite Database', 'Azure SQL Database',
     'Relational integrity, automated backups, enterprise compliance'],
    ['Pandas Pipeline', 'Azure Databricks + Data Factory',
     'Databricks handles PySpark workloads; ADF orchestrates triggers'],
    ['File Storage (local)', 'Azure Data Lake Storage Gen2',
     'Hierarchical namespace: Raw, Staged, Curated zones'],
    ['Power BI CSV Export', 'Microsoft Fabric (DirectLake)',
     'Power BI connects directly to Parquet files in OneLake'],
    ['LangChain + Groq LLM', 'Azure OpenAI Service',
     'Enterprise SLAs, data privacy, RBAC controls'],
    ['ChromaDB (Local RAG)', 'Azure AI Search',
     'Built-in semantic ranking, hybrid search, Azure OpenAI integration'],
    ['Local bcrypt secrets', 'Azure Key Vault',
     'Centralized secret management with role-based access'],
    ['Local Docker build', 'Azure Container Registry (ACR)',
     'Secure private image registry integrated with App Service'],
]
story.append(tbl(arch_data, col_widths=[4.5*cm, 5.5*cm, 8*cm]))
story.append(sp(8))

story.append(Paragraph('Data Engineering Pipeline', H2))
for step, detail in [
    ('Step 1 — Ingest', 'Raw CSV files (train.csv, synthetic versions) are uploaded to Azure Data Lake Gen2 in the raw/ container.'),
    ('Step 2 — Orchestrate', 'Azure Data Factory triggers a pipeline on schedule or event (new file arrival).'),
    ('Step 3 — Transform', 'Azure Databricks runs PySpark jobs to clean, validate, and enrich data — handling null values, type casting, date parsing, and feature engineering.'),
    ('Step 4 — Store', 'Curated Parquet files are written to the curated/ container in ADLS Gen2 with optimized partitioning (by region/year/category).'),
    ('Step 5 — Serve', 'Microsoft Fabric connects via DirectLake to Parquet files — enabling real-time Power BI dashboards without data movement.'),
]:
    story.append(Paragraph(f'<b>{step}:</b> {detail}', ParagraphStyle(
        'step', fontSize=10, leading=14, spaceBefore=4, spaceAfter=4, leftIndent=12)))

story.append(Paragraph('Multi-Agent AI Architecture', H2))
story.append(Paragraph(
    'The AI layer uses LangChain as the orchestration framework coordinating multiple specialized agents. '
    'The Retrieval-Augmented Generation (RAG) system uses Azure AI Search as the vector store, indexed '
    'with product knowledge, policy documents, and historical order data. Azure OpenAI GPT-4o is the '
    'inference engine powering all natural language responses.', BODY))

for agent, role in [
    ('Query Router Agent', 'Classifies incoming user questions and routes them to the appropriate specialized agent or data source.'),
    ('Analytics Agent', 'Handles quantitative questions — sales figures, aggregations, comparisons, and trend analysis.'),
    ('Product Agent', 'Answers product catalog questions — descriptions, categories, pricing, and availability.'),
    ('Customer Agent', 'Handles customer profile lookups, order history, and segment-level insights.'),
    ('Logistics Agent', 'Answers shipping-related questions — delivery times, ship modes, and regional policies.'),
    ('RAG Knowledge Agent', 'Retrieves from the vector store (this document) for policy, architecture, and general knowledge questions.'),
]:
    story.append(bullet(f'<b>{agent}:</b> {role}'))
story.append(sp(6))

story.append(Paragraph('CI/CD Pipeline', H2))
for step in [
    'GitHub Actions builds the Docker image on every push to main branch.',
    'Image is pushed to Azure Container Registry (ACR) with versioned tags.',
    'A webhook is triggered on Azure App Service to pull the latest image automatically.',
    'Environment variables and API keys are stored in Azure Key Vault and injected at runtime.',
    'Database migrations are run automatically as part of the deployment pipeline.',
]:
    story.append(bullet(step))

story.append(Paragraph('Migration Checklist', H2))
for item in [
    'Provision Azure SQL Database and run migration script for users and uploads tables.',
    'Upload train.csv to ADLS Gen2 raw/ container.',
    'Migrate data_pipeline.py logic to a Databricks Notebook.',
    'Provision Azure OpenAI and update .env or Key Vault with AZURE_OPENAI_API_KEY.',
    'Push local Docker image to Azure Container Registry.',
    'Connect Power BI Service directly to the Microsoft Fabric workspace.',
    'Configure Azure AI Search index with product and knowledge base documents.',
    'Set up Azure Key Vault references in App Service configuration.',
]:
    story.append(bullet(f'☐ {item}'))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 11 — FAQ
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('11. Frequently Asked Questions (100+ Questions)', H1))
story.append(hr())
story.append(Paragraph(
    'This section covers the most likely questions users will ask the RetailGPT chatbot, '
    'organized by topic. Answers are grounded in actual dataset statistics.', BODY))

faq_sections = [

    ('General Business Questions', [
        ('What is this retail store called?', 'The store is referred to as the Superstore — a fictional US-based retail company used for business intelligence and analytics training. It sells products in three main categories: Furniture, Office Supplies, and Technology.'),
        ('What does this store sell?', 'The Superstore sells products in three categories: (1) Furniture — chairs, tables, bookcases, furnishings; (2) Office Supplies — binders, paper, storage, art supplies, appliances, labels, envelopes, fasteners; (3) Technology — phones, accessories, machines, copiers.'),
        ('How many products does the store carry?', 'The store carries 1,849 unique products across 17 sub-categories.'),
        ('What is the total revenue of the store?', f'Total revenue from the dataset is ${total_sales:,.2f} across 9,800 transactions.'),
        ('How many orders have been placed?', f'There are {total_orders:,} unique orders in the dataset, from a total of 9,800 transaction line items.'),
        ('How many customers does the store have?', f'The store has {total_cust} unique customers in the dataset.'),
        ('In which country does the store operate?', 'The Superstore operates exclusively in the United States, serving all four major US geographic regions.'),
        ('What time period does the data cover?', 'The dataset covers sales from 2015 through 2018, spanning approximately 4 years of transactions.'),
        ('What is the average order value?', f'The average revenue per transaction line item is ${df["Sales"].mean():,.2f}. The average total order value (sum of all items per order) is approximately ${avg_order:,.2f}.'),
        ('What was the highest single sale ever?', f'The highest single transaction was ${df["Sales"].max():,.2f} — a Technology product ordered by Sean Miller (Order ID: CA-2015-145317).'),
    ]),

    ('Product & Category Questions', [
        ('What are the main product categories?', 'The three main product categories are: (1) Furniture, (2) Office Supplies, and (3) Technology.'),
        ('Which category generates the most revenue?', 'Technology generates the most revenue at $827,455 (36.8% of total), despite having the fewest transactions (1,813). Office Supplies is second at $705,422, and Furniture third at $728,658.'),
        ('Which category has the most orders?', 'Office Supplies has by far the most orders — 5,909 transactions, representing 60.3% of all orders. It is the high-volume, lower-value category.'),
        ('What is the most expensive product category?', 'Technology has the highest average sale value at $456.40 per transaction, making it the most expensive category on average.'),
        ('What are all 17 sub-categories?', 'The 17 sub-categories are: Bookcases, Chairs, Furnishings, Tables (Furniture); Art, Binders, Envelopes, Fasteners, Labels, Paper, Storage, Supplies, Appliances (Office Supplies); Accessories, Copiers, Machines, Phones (Technology).'),
        ('Which sub-category generates the most revenue?', 'Phones generates the most sub-category revenue at $327,782. Chairs is close behind at $322,822.'),
        ('Which sub-category has the highest average sale?', 'Copiers have the highest average sale at $2,215.88 per transaction, followed by Machines at $1,645.55.'),
        ('Which sub-category is ordered most frequently?', 'Binders (1,492 orders) and Paper (1,338 orders) are the most frequently ordered sub-categories.'),
        ('What is the cheapest sub-category on average?', 'Fasteners are the cheapest, averaging only $14.03 per transaction.'),
        ('How many furniture products are available?', 'Furniture contains approximately 550 unique products across Chairs (88), Tables (56), Bookcases (50), and Furnishings (186 products).'),
        ('What types of chairs are sold?', 'The Chairs sub-category includes executive leather chairs, ergonomic task chairs, folding chairs, conference room chairs, and drafting chairs from brands like Global, Hon, and Situation.'),
        ('What technology products are sold?', 'Technology products include smartphones, desk phones, cordless phones, computer accessories (mice, keyboards, headsets, webcams), copiers, fax machines, label printers, and binding equipment.'),
        ('What is the most ordered product?', 'Staple envelope is the most frequently ordered product with 47 orders. Staples (46) and Easy-staple paper (44) are close second and third.'),
        ('How many sub-categories does Office Supplies have?', 'Office Supplies has 9 sub-categories: Art, Binders, Envelopes, Fasteners, Labels, Paper, Storage, Supplies, and Appliances.'),
        ('Which product has the most SKUs available?', 'Paper has the widest selection with 277 unique product variants, followed by Binders (211 products) and Phones (189 products).'),
    ]),

    ('Sales & Revenue Questions', [
        ('What is the total revenue?', f'Total revenue is ${total_sales:,.2f} across all 9,800 transaction records.'),
        ('What is the median sale value?', f'The median (50th percentile) sale value is ${df["Sales"].median():,.2f}.'),
        ('What is the minimum sale value ever recorded?', f'The minimum sale value is ${df["Sales"].min():,.4f}.'),
        ('What is the maximum sale value ever recorded?', f'The maximum sale value is ${df["Sales"].max():,.2f}.'),
        ('Which region has the highest sales?', f'The West region leads with ${710219.68:,.2f} in total revenue (31.6% of total), followed by the East at ${669518.73:,.2f}.'),
        ('Which region has the most orders?', 'The West has the most orders with 3,140 transactions (32.0%), followed by the East with 2,785 orders.'),
        ('Which state generates the most revenue?', 'California is the top state at $446,306 in revenue — nearly 1.5x more than second-place New York ($306,361).'),
        ('What are the top 5 states by sales?', 'Top 5 states: (1) California $446,306, (2) New York $306,361, (3) Texas $168,572, (4) Washington $135,207, (5) Pennsylvania $116,277.'),
        ('Which city has the highest sales?', 'New York City leads all cities with $252,462 in total revenue, followed by Los Angeles at $173,420.'),
        ('What percentage of revenue comes from Technology?', f'Technology accounts for approximately 36.8% of total revenue (${827455.87:,.2f} of ${total_sales:,.2f}).'),
        ('What percentage of orders are Office Supplies?', 'Office Supplies accounts for 60.3% of all transaction orders (5,909 out of 9,800).'),
        ('How much does the average customer spend?', f'Across all customers, total revenue is distributed across {total_cust} customers, averaging approximately ${total_sales/total_cust:,.2f} per customer.'),
        ('What is the revenue split by segment?', 'Consumer: $1,148,061 (51.1%), Corporate: $688,494 (30.6%), Home Office: $424,982 (18.9%).'),
    ]),

    ('Customer Questions', [
        ('Who is the top customer by revenue?', 'Sean Miller is the top customer with $25,043 in total purchases — primarily from a single large Technology order ($22,638).'),
        ('Who are the top 5 customers?', 'Top 5 customers by revenue: (1) Sean Miller $25,043, (2) Tamara Chand $19,052, (3) Raymond Buch $15,117, (4) Tom Ashbrook $14,596, (5) Adrian Barton $14,474.'),
        ('How many customer segments are there?', 'There are three customer segments: Consumer, Corporate, and Home Office.'),
        ('Which segment has the most customers?', 'The Consumer segment has the most transactions (5,101 = 52%) and represents the largest customer group.'),
        ('Which segment spends the most per transaction?', 'Home Office has the highest average transaction value at $243.40, slightly above Corporate ($233.15) and Consumer ($225.07).'),
        ('Who orders most frequently?', 'William Brown orders most frequently with 35 transactions, followed by Matt Abelman and Paul Prost (34 each).'),
        ('How many unique customers are there?', f'There are {total_cust} unique customers in the dataset.'),
        ('What is a Corporate segment customer?', 'Corporate customers are businesses purchasing for organizational needs. They represent 30.6% of revenue and tend to make slightly higher-value purchases than Consumer segment.'),
        ('What is a Home Office customer?', 'Home Office customers are remote workers and small business owners. They make up 18.9% of revenue and have the highest average purchase value ($243) among segments.'),
    ]),

    ('Shipping & Logistics Questions', [
        ('What shipping options are available?', 'Four shipping modes are available: Standard Class (economy), Second Class (standard), First Class (expedited), and Same Day (premium instant delivery).'),
        ('How long does Standard Class shipping take?', 'Standard Class takes 3–7 business days, with an average of 5.0 days.'),
        ('How long does First Class shipping take?', 'First Class takes 1–4 business days, with an average of 2.2 days.'),
        ('How long does Same Day shipping take?', 'Same Day delivery is completed the same day or within 1 business day (average 0.04 days from order to ship).'),
        ('How long does Second Class shipping take?', 'Second Class takes 1–5 business days, with an average of 3.2 days.'),
        ('What is the most popular shipping method?', 'Standard Class is most popular with 5,859 orders (59.8% of all shipments).'),
        ('How many orders use Same Day shipping?', 'Only 538 orders (5.5% of all orders) use Same Day shipping — it is the least common option.'),
        ('Is shipping available across all US states?', f'Yes, the Superstore ships to {df["State"].nunique()} US states and {df["City"].nunique()} cities nationwide.'),
        ('Does shipping mode affect what I can buy?', 'No — all shipping modes are available for all products. The choice affects delivery speed and cost, not product eligibility.'),
        ('What is the fastest shipping option?', 'Same Day is the fastest option, completing delivery the same day as order placement in most cases.'),
    ]),

    ('Technical / System Questions', [
        ('What is RetailGPT?', 'RetailGPT is an AI-powered retail intelligence system built with LangChain, FastAPI, Streamlit, and Azure OpenAI. It answers natural language questions about Superstore sales data using a multi-agent RAG architecture.'),
        ('What AI model powers RetailGPT?', 'RetailGPT uses Azure OpenAI GPT-4o as the primary inference model, with LangChain as the orchestration framework and Azure AI Search as the vector database for RAG retrieval.'),
        ('What is a RAG system?', 'RAG (Retrieval-Augmented Generation) is an AI architecture that retrieves relevant documents from a knowledge base and injects them into the LLM prompt to produce grounded, accurate answers.'),
        ('Where is the data stored?', 'In production Azure deployment, data is stored in Azure Data Lake Gen2 as Parquet files. Customer/user data is in Azure SQL Database. The vector index is in Azure AI Search.'),
        ('How is user authentication handled?', 'Locally, SQLite stores users with bcrypt password hashing. In Azure production, users are stored in Azure SQL Database and secrets managed via Azure Key Vault.'),
        ('What database does RetailGPT use?', 'The application uses SQLite locally for development and Azure SQL Database in production. Data is served from Azure Data Lake (Parquet) for analytics.'),
        ('How is the data pipeline structured?', 'The pipeline flows: Raw CSV → Azure Data Lake Gen2 raw/ → Azure Data Factory orchestration → Azure Databricks transformation → Curated Parquet → Microsoft Fabric / Power BI.'),
        ('What is the frontend built with?', 'The frontend is built with Streamlit, deployed as an Azure App Service container.'),
        ('What is the backend built with?', 'The backend uses FastAPI with multiple routers, containerized with Docker and deployed on Azure App Service.'),
        ('How does the multi-agent system work?', 'A query router agent classifies incoming questions and routes them to specialized agents: Analytics Agent, Product Agent, Customer Agent, Logistics Agent, and RAG Knowledge Agent.'),
        ('What datasets are used for training?', 'Three CSV files: train.csv (original 9,800 rows), synthetic_balanced_v1.csv (9,800 rows, balanced), and synthetic_balanced_v2.csv (9,800 rows, higher-value synthetic data).'),
        ('What is synthetic_balanced_v1.csv?', 'synthetic_balanced_v1.csv is a synthetically generated dataset with 9,800 rows designed to balance sub-category representation. It has a higher average sale ($2,044) than the original train.csv ($230).'),
        ('What is the difference between train.csv and synthetic versions?', 'train.csv is the original real-world dataset with natural class imbalance (60% Office Supplies). Synthetic versions artificially balance sub-categories for better ML model training.'),
        ('How is the Azure deployment done?', 'GitHub Actions builds Docker images, pushes to Azure Container Registry, and triggers automatic deployment via webhook to Azure App Service.'),
        ('What is DirectLake in the architecture?', 'DirectLake is a Microsoft Fabric feature that allows Power BI to connect directly to Parquet files in OneLake (ADLS Gen2) without importing data — enabling real-time analytics at scale.'),
    ]),

    ('Analytical & Business Intelligence Questions', [
        ('What is the most profitable sub-category?', 'Without profit margin data, we use revenue as a proxy. Phones ($327,782) generates the most sub-category revenue. However, Copiers have the highest value per sale ($2,215) suggesting high margins.'),
        ('What products are frequently bought together?', 'While basket analysis requires transaction-level join analysis, Staple envelopes, Staples, and Easy-staple paper are frequently co-ordered — typical office supply replenishment bundles.'),
        ('What are the highest-risk customer accounts to lose?', 'Top customers Sean Miller ($25,043), Tamara Chand ($19,052), and Raymond Buch ($15,117) represent highest revenue-at-risk. High-frequency buyers like William Brown (35 orders) are also critical for recurring revenue.'),
        ('Which region is underperforming?', 'The South region has the fewest orders (1,598) and lowest total revenue ($389,151 = 17.3%) despite having the highest average sale ($243.52). This suggests fewer customers but high-value purchases when they occur.'),
        ('What is the revenue concentration risk?', 'California alone accounts for 19.8% of total revenue. The top 2 states (California and New York) contribute 33.5% of revenue. High geographic concentration is a business risk.'),
        ('How does Home Office segment differ from Consumer?', 'Home Office customers have a higher average purchase value ($243 vs $225) but represent only 18.9% of revenue vs 51.1% for Consumer. Home Office buyers make fewer but higher-value purchases.'),
        ('Are there seasonal sales patterns?', 'Without aggregating by month/quarter, the dataset covers 2015–2018. Typical retail seasonal patterns (Q4 holiday peaks) would need time-series analysis to confirm in this dataset.'),
        ('What is the customer lifetime value metric?', f'Average customer lifetime value in this dataset is approximately ${total_sales/total_cust:,.2f} (total revenue ÷ unique customers). Top customers like Sean Miller far exceed this at $25,043.'),
        ('Which products have the highest unit value?', 'Copiers average $2,215.88 per order, Machines $1,645.55, Tables $645.89, and Chairs $531.83 are the highest unit value sub-categories.'),
        ('What percentage of revenue comes from the top 10 customers?', f'Top 10 customers together contribute approximately $152,225 out of ${total_sales:,.2f} total — roughly 6.8% of all revenue from just 10 customers.'),
    ]),

    ('Operational Questions', [
        ('How do I filter orders by region?', 'The Region column contains four values: East, West, Central, South. Filter by this column to analyze regional performance.'),
        ('How do I identify high-value orders?', 'Sort or filter the Sales column for values above $1,000. The top single transactions are in the Technology category (Copiers, Machines, Phones).'),
        ('Can I search by customer name?', 'Yes — the Customer Name column contains full names (e.g., "Sean Miller"). Customer IDs follow the format XX-##### (e.g., CG-12520).'),
        ('How are orders identified?', 'Orders use the format XX-YYYY-######. CA prefix means California (domestic), US prefix may indicate a different origin state. Example: CA-2017-152156.'),
        ('How are products identified?', 'Products have a Product ID in format CAT-SUBCAT-####### (e.g., FUR-BO-10001798 = Furniture-Bookcases). Product Names are full descriptive strings.'),
        ('What date format is used?', 'Dates are in DD/MM/YYYY format (day first). Example: 08/11/2017 = November 8, 2017.'),
        ('How many states does the store ship to?', f'The store ships to {df["State"].nunique()} states across the continental United States.'),
        ('What is the postal code format?', 'Postal codes are standard US ZIP codes (5-digit numeric). Some synthetic dataset versions may include decimal points (e.g., 49505.0) which should be cleaned.'),
        ('Can I filter by product category?', 'Yes — the Category column contains exactly three values: Furniture, Office Supplies, Technology. The Sub-Category column provides 17 more granular groupings.'),
        ('What is the Row ID field?', 'Row ID is a simple sequential integer (1, 2, 3...) that uniquely identifies each transaction row. It is not an order or product identifier.'),
    ]),
]

for section_title, questions in faq_sections:
    story.append(Paragraph(section_title, H2))
    for q, a in questions:
        story.extend(qa(q, a))
    story.append(thin_hr())
    story.append(sp(4))

story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════
# SECTION 12 — GLOSSARY
# ═══════════════════════════════════════════════════════════════════
story.append(Paragraph('12. Glossary of Terms', H1))
story.append(hr())

glossary = [
    ['Term', 'Definition'],
    ['RAG', 'Retrieval-Augmented Generation — AI architecture that combines vector search with LLM generation for grounded answers.'],
    ['LangChain', 'Python framework for building LLM-powered applications with chains and agents.'],
    ['Azure OpenAI', 'Microsoft Azure-hosted version of OpenAI GPT models with enterprise SLAs and data privacy.'],
    ['Azure AI Search', 'Microsoft Azure vector database and semantic search service, used as the RAG vector store.'],
    ['ChromaDB', 'Open-source local vector database used in development mode before Azure AI Search deployment.'],
    ['FastAPI', 'High-performance Python web framework used for the RetailGPT backend API.'],
    ['Streamlit', 'Python framework for building interactive web applications, used as the RetailGPT frontend.'],
    ['ADLS Gen2', 'Azure Data Lake Storage Generation 2 — hierarchical cloud storage for big data workloads.'],
    ['Azure Databricks', 'Managed Apache Spark service on Azure used for large-scale data transformation.'],
    ['Azure Data Factory', 'Cloud ETL/ELT orchestration service that triggers and manages data pipelines.'],
    ['Microsoft Fabric', 'Unified analytics platform from Microsoft combining data engineering, warehousing, and Power BI.'],
    ['DirectLake', 'Fabric/Power BI mode that reads directly from Delta/Parquet files in OneLake without import.'],
    ['Transaction', 'A single line item in an order — one product purchased at a specific quantity and price.'],
    ['Order', 'A group of transactions (line items) placed together with a single Order ID.'],
    ['SKU', 'Stock Keeping Unit — a unique identifier for a product variant (mapped to Product ID here).'],
    ['Segment', 'Customer classification: Consumer, Corporate, or Home Office based on buyer type.'],
    ['Sub-Category', 'A specific product type within a category (e.g., Chairs within Furniture).'],
    ['Ship Mode', 'The delivery speed option selected: Standard Class, Second Class, First Class, Same Day.'],
    ['Region', 'US geographic division: East, West, Central, or South.'],
    ['bcrypt', 'Password hashing algorithm used to securely store user passwords.'],
    ['Azure Key Vault', 'Azure service for securely storing and accessing secrets, keys, and certificates.'],
    ['ACR', 'Azure Container Registry — private Docker image registry for containerized deployments.'],
    ['CI/CD', 'Continuous Integration / Continuous Deployment — automated build, test, and deploy pipeline.'],
    ['Vector Store', 'Database optimized for storing and querying high-dimensional embedding vectors for semantic search.'],
    ['Embedding', 'Numerical vector representation of text that captures semantic meaning for similarity search.'],
    ['GPT-4o', 'OpenAI\'s multimodal language model used as the inference engine for RetailGPT responses.'],
    ['Parquet', 'Columnar storage file format optimized for analytics workloads; used in the curated data layer.'],
    ['PySpark', 'Python API for Apache Spark, used in Databricks for large-scale parallel data processing.'],
    ['Webhook', 'HTTP callback mechanism used to trigger Azure App Service redeployment after ACR image push.'],
]
story.append(tbl(glossary, col_widths=[4.5*cm, 13.5*cm]))
story.append(sp(12))

story.append(thin_hr())
story.append(Paragraph(
    'RetailGPT RAG Knowledge Base | Version 1.0 | Built from Superstore Transactional Data | '
    'This document is auto-generated and designed for ingestion into a RAG vector store.',
    ParagraphStyle('end', fontSize=8, textColor=MUTED, alignment=TA_CENTER)))

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f'PDF written to: {OUTPUT}')
