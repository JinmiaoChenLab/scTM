"""Generate the code api pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("sctm").rglob("*.py")):

    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to("sctm").with_suffix(".md")
    full_doc_path = Path("api", doc_path)
    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        continue
    # parts = parts[:-1]
    # doc_path = doc_path.with_name("index.md")
    # full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if parts[-1] == "stamp":
        nav[parts] = doc_path.as_posix()  #
    else:
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:   
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:  # 
    print("opened")
    nav_file.writelines(nav.build_literate_nav())  # 