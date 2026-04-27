"""
Configuration constants and plasmid name mappings for COF analysis.

This module centralizes all experiment-specific mappings to avoid duplication
across notebooks and ensure consistency.
"""

# Full descriptive name mapping for reporter plasmids
# ⚠️ Key order determines default plotting order via plasmid_order
REPORTER_PLASMID_NAME_MAPPING = {
    # sfGFP series (co-translational folding)
    'pNZ212': '6sfGFP',           # 6x sfGFP, 0x mCh
    'pRS026': '5sfGFP-1mCh',      # 5x sfGFP, 1x mCh
    'pRS027': '4sfGFP-2mCh',      # 4x sfGFP, 2x mCh
    'pRS028': '3sfGFP',           # 3x sfGFP, 0x mCh (NO mCherry!)
    'pRS029': '3sfGFP-3mCh',      # 3x sfGFP, 3x mCh
    'pRS030': '2sfGFP-4mCh',      # 2x sfGFP, 4x mCh
    'pRS031': '1sfGFP-5mCh',      # 1x sfGFP, 5x mCh
    'PNZ405': '6xsfGFP (trunc)',  # 6x sfGFP (truncated), 0x mCh

    # mGFPuv series (slow folding GFP)
    'pNZ370': '6GFPuv',          # 6x mGFPuv, 0x mCh
    'pRS032': '4GFPuv-2mCh',     # 4x mGFPuv, 2x mCh
    'pNZ396': '3GFPuv-3mCh',     # 3x mGFPuv, 3x mCh
    'pNZ389': '2GFPuv-4mCh',     # 2x mGFPuv, 4x mCh
    'pNZ388': '1GFPuv-5mCh',     # 1x mGFPuv, 5x mCh

    # XBP1 variants - C-terminal XBP1(S255A)
    'pRS038': '4sfGFP-2mCh-Xbp1',    # 4x sfGFP, 2x mCh, Xbp1(S255A) at C-term
    'pRS039': '4GFPuv-2mCh-Xbp1',   # 4x mGFPuv, 2x mCh, Xbp1(S255A) at C-term

    # XBP1 variants - N-terminal XBP1(S255A)
    'pRS048': 'Xbp1-4sfGFP-2mCh',    # Xbp1(S255A) at N-term, 4x sfGFP, 2x mCh
    'pRS049': 'Xbp1-4GFPuv-2mCh',   # Xbp1(S255A) at N-term, 4x mGFPuv, 2x mCh

    # Deoptimized variants (same GFP copy number but with deoptimized codons)
    'pRS084': '4sfGFP-deopt-2mCh',   # 4x sfGFP (deoptimized), 2x mCh
    'pRS045': '4GFPuv-deopt-2mCh',  # 4x mGFPuv (deoptimized), 2x mCh

    # mCherry-only control (always plotted last)
    'pNZ381': '6mCh',            # 0x sfGFP, 6x mCh

    # Reporter plasmid (Halo-based detection)
    # 'pRS012': 'Halo-reporter',      # piggybac-Tet-on-anti-HAfb-Halo-IRES-LaG16-tdStayGold
}

# Short name mapping for individual day naming (based on sfGFP/uvGFP count)
PLASMID_SHORT_NAME_MAPPING = {
    'pNZ212': '6sf',   # 6 sfGFP copies
    'pRS026': '5sf',   # 5 sfGFP copies
    'pRS027': '4sf',   # 4 sfGFP copies
    'pRS028': '3sf',   # 3 sfGFP copies (no mCh)
    'pRS029': '3sf',   # 3 sfGFP copies
    'pRS030': '2sf',   # 2 sfGFP copies
    'pRS031': '1sf',   # 1 sfGFP copy
    'PNZ405': '6sf-trc', # 6 sfGFP
    'pNZ370': '6uv',   # 6 GFPuv copies
    'pRS032': '4uv',   # 4 GFPuv copies
    'pNZ396': '3uv',   # 3 GFPuv copies
    'pNZ389': '2uv',   # 2 GFPuv copies
    'pNZ388': '1uv',   # 1 GFPuv copy
    'pRS038': '4sf-Xbp1',   # 4 sfGFP + Xbp1
    'pRS039': '4uv-Xbp1',   # 4 GFPuv + Xbp1
    'pRS048': 'Xbp1-4sf',   # Xbp1 + 4 sfGFP
    'pRS049': 'Xbp1-4uv',   # Xbp1 + 4 GFPuv
    'pRS084': '4sf-deopt',   # 4 sfGFP (deoptimized)
    'pRS045': '4uv-deopt',   # 4 GFPuv (deoptimized)
    'pNZ381': '0sf',   # 0 sfGFP copies (mCh-only control)
}
