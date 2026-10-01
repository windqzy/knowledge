current_title = "#标题1"
current_lines:list[str] = ["#标题1","内容1","内容2"]
current_line_strip = "#标题2"


current_title = current_title + "\n" + current_line_strip
# 标题2
title_list: list[str] = current_title.split("\n")
title_list.extend(current_lines[len(title_list) - 1:])
current_lines = title_list

print(current_title)
print(current_lines)
