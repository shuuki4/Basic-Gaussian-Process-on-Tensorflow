
function max(a, b) {
	return a>b?a:b;
};

function min(a,b) {
	return a<b?a:b;
}

function drawGraph(train, result) {
	var xScale = d3.scaleLinear().domain([0.0, 7.5]).range([30, 480]);
	var yScale = d3.scaleLinear().domain([0, 5]).range([480, 30]);
	yAxis = d3.axisLeft().scale(yScale).ticks(8).tickSize(5);
	xAxis = d3.axisBottom().scale(xScale).tickSize(5).tickValues([1,2,3,4,5,6,7])
	d3.select("svg").append("g").attr("transform", "translate(30, 0)")
	.attr("id", "yAxisG").call(yAxis);
	d3.select("svg").append("g").attr("transform", "translate(0, 480)")
	.attr("id", "xAxisG").call(xAxis);

	var originalLine = d3.line().x(function(d) { return xScale(d.x); })
	.y(function(d) { return yScale(d.y); }).curve(d3.curveBasis);
	d3.select("svg").append("path").attr("d", originalLine(train)).attr("class", "original")
	.attr("fill", "none").attr("stroke", "black").attr("stroke-width", 1)
	
	d3.select("svg").selectAll("circle").data(result).enter()
	.append("circle")
	.attr("r", 4).attr("cx", function(d) { return xScale(d.x); }).attr("cy", function(d) { return yScale(d.y); })
	.style("fill", "black").style("opacity", .5);
	
	var varArea = d3.area()
	.x(function(d) { return xScale(d.x); })
	.y1(function(d) { return yScale(min(parseFloat(d.y)+parseFloat(d.sig)*1.96, 5.0)); })
	.y0(function(d) { return yScale(max(parseFloat(d.y)-parseFloat(d.sig)*1.96, 0.0)); })
	.curve(d3.curveCardinal);
	
	d3.select("svg").append("path").attr("d", varArea(result)+"Z")
	.attr("fill", "red").style("opacity", .5);
	
};

function jsOnLoad() {
	d3.queue()
	.defer(d3.csv, "train.csv")
	.defer(d3.csv, "result.csv")
	.await(function(error, file1, file2) {
		drawGraph(file1, file2);
	});
}