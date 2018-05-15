var data;
var csv_path = "titanic.csv";

// 观察视角。看百分比或绝对数量.默认是绝对数量.
var percent = 10;
var absolute = 11;
var observation = absolute;

function set_observation(obs) {
    observation = obs;
}

// 保存加载的数据.
d3.csv(csv_path, function (d) {
    data = d;
});

// 使用dimple绘制图表
function draw(data, filter_value) {
    "use strict";
    var margin = 75,
        width = 1400 - margin,
        height = 600 - margin;

    var svg = d3.select("body")
        .append("svg")
        .attr("width", width + margin)
        .attr("height", height + margin)
        .append('g')
        .attr('class', 'chart');

    if (filter_value != "All") {
        data = dimple.filterData(data, "Type", filter_value);
    }

    var myChart = new dimple.chart(svg, data);
    var axis_x = myChart.addCategoryAxis("x", "Class");
    axis_x.addOrderRule(["1st class", "2nd class", "3rd class"]);

    // 根据observation设置y轴以绝对数量显示或者以百分比显示
    if (observation === percent) {
        myChart.addPctAxis("y", "Count");
    } else {
        myChart.addMeasureAxis("y", "Count");
    }
    var series = myChart.addSeries("Survived", dimple.plot.bar);
    series.addOrderRule(["Perished", "Survived"]);
    series.addEventHandler("click", function (event) {
        var obs;
        if (observation == percent) {
            obs = " in Percent";
        } else {
            obs = " in Absolute";
        }
        alert("当前查看类型:" + filter_value + obs);
    });
    myChart.assignColor("Perished", "#A82A0D", "black", 0.7);
    myChart.assignColor("Survived", "#0DDF30", "black", 0.7);
    // 添加说明
    myChart.addLegend(150, 25, width, height, "left", series);
    myChart.draw();
}

// 按钮调用的绘制方法。type: 要过滤的类型.
function draw_chart(type) {
    var filter_value;
    switch (type) {
        case 0:
        default:
            filter_value = "All";
            break;
        case 1:
            filter_value = "Men";
            break;
        case 2:
            filter_value = "Women";
            break;
        case 3:
            filter_value = "Children";
            break;
    }

    // 从body中去除上次添加的svg。
    d3.select("body").select("svg").remove();

    // 确保data已经加载完成。
    if (data != null) {
        draw(data, filter_value);
    }
    else {
        d3.csv(csv_path, draw);
    }
}