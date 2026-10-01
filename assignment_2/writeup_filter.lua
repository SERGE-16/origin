-- Keep PDF-specific formatting out of assignment_2_WRITEUP.md.

function RawBlock(block)
    if block.format == "html"
        and block.text:match("^%s*<!%-%-%s*pagebreak%s*%-%->%s*$")
        and FORMAT:match("latex")
    then
        return pandoc.RawBlock("latex", "\\clearpage")
    end
end

function Image(image)
    if not FORMAT:match("latex") then
        return
    end

    if image.src:match("LEGGED_ROBOTS_ASSIGNMENT_2_SKETCHES") then
        image.attributes.width = "60%"
    elseif image.src:match("roa_map") or image.src:match("image%%20copy") then
        image.attributes.width = "78%"
    else
        image.attributes.width = "82%"
    end

    return image
end
