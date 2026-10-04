"""Fixed current-frame spatial selection for the proposed method's first pilot.

This deliberately simple selector is not a novelty claim or a replacement for
the published comparator. Return absolute visual positions for the qualified
compaction map, preserving all nonvisual positions separately.
"""


def stratified_visual_positions(budget):
    if type(budget) is not int or budget not in (256,384,512):
        raise ValueError("frozen diagnostic budgets are 256, 384 or dense 512")
    per_cell=budget//128
    selected=[]
    for camera in range(2):
        for row in range(0,16,2):
            for column in range(0,16,2):
                # Opposite corners first; rotation distributes the omitted
                # corner for 3/4 retention. No result-dependent selection.
                corners=((0,0),(1,1),(0,1),(1,0))
                shift=(row//2+column//2+camera)%4
                if per_cell==2:
                    chosen=((0,0),(1,1)) if shift%2==0 else ((0,1),(1,0))
                else:
                    chosen=tuple(corners[(shift+i)%4] for i in range(per_cell))
                selected.extend(1+256*camera+16*(row+dr)+column+dc for dr,dc in chosen)
    return tuple(sorted(selected))
