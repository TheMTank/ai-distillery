array_objs = []

$.each($('.col-md-12'), function(index, el) {
    array_objs.push({
        "title": $(el).find('h5').text().trim(),
        "tldr": $(el).find('mb-0').text().trim(),
        "abstract": $(el).find('i').text().trim()
    })
})

console.log($(el).text().trim());
