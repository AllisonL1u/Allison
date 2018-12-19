/**
 * Created by Allison on 2018/12/19.
 */
 function formatNum(money) {
    for (var a = ""; money > 999;) {
        var s = money.toString();
        a = "," + s.substring(s.length - 3) + a;
        money = parseInt(money / 1000)
    }

    return money + a;
}

function showCurrentTime() {
    var time = new Date;
    var time = time.toLocaleDateString() + time.toLocaleTimeString();
    document.getElementById("time").innerHTML = time
}

function loadData() {
	$.ajax({
		url: "http://cn.bing.com",
        type: "GET",
        dataType: 'JSON',
        success: function (result) {
        	result['nano_1'] = parseInt(result['aaa'])
        	result['nano_2'] = parseInt(result['aaa'])
        	result['nano_3'] = parseInt(result['aaa'])
        	result['15_days_vip_free'] = parseInt(result['aaa'])
        	result['vip_purchase'] = parseInt(result['aaa'])
        	result['vip_gmv'] = parseInt(result['aaa'])
        	result['album_purchase_gmv'] = parseInt(result['aaa'])
        	result['kol_course_pv'] = parseInt(result['aaa'])
        	result['kol_course_uv'] = parseInt(result['aaa'])
        	result['lunch_pv'] = parseInt(result['aaa'])
        	result['lunch_uv'] = parseInt(result['aaa'])
        	for (var key in result) {
        		$("." + key).text(formatNum(result[key]));
                $("." + key).css("font-size","small");
        	}
        }
	});
	$(.nano_1).text(formatNum(999999999999));
    $(.nano_1).css("font-size","small");
    $(.nano_2).text(formatNum(999999999999));
    $(.nano_2).css("font-size","small");
    $(.nano_3).text(formatNum(999999999999));
    $(.nano_3).css("font-size","small");
    $(.15_days_vip_free).text(formatNum(999999999999));
    $(.15_days_vip_free).css("font-size","small");
    $(.vip_purchase).text(formatNum(999999999999));
    $(.vip_purchase).css("font-size","small");
    $(.vip_gmv).text(formatNum(999999999999));
    $(.vip_gmv).css("font-size","small");
    $(.album_purchase_gmv).text(formatNum(999999999999));
    $(.album_purchase_gmv).css("font-size","small");
    $(.kol_course_pv).text(formatNum(999999999999));
    $(.kol_course_pv).css("font-size","small");
    $(.kol_course_uv).text(formatNum(999999999999));
    $(.kol_course_uv).css("font-size","small");
    $(.lunch_pv).text(formatNum(999999999999));
    $(.lunch_pv).css("font-size","small");
    $(.lunch_uv).text(formatNum(999999999999));
    $(.lunch_uv).css("font-size","small");
}

showCurrentTime();
setInterval(showCurrentTime, 1000);
loadData();
setInterval(loadData, 5000);

