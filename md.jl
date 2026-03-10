# Abstract hierarchy
abstract type GeometricObject end
abstract type Shape{T <: Number} <: GeometricObject end

# Concrete shapes
struct Circle{T <: Number} <: Shape{T}
    radius::T
end

struct Rectangle{T <: Number} <: Shape{T}
    width::T
    height::T
end
struct Triangle{T <: Number} <: Shape{T}
    base::T
    height::T
end
# Area implementations

area(c::Circle) = π * c.radius^2

area(r::Rectangle) = r.width * r.height

function area(r::Rectangle{Float64}) 
    println("Calculating area for a rectangle with Float64 precision")
    return r.width * r.height
end

# Generic fallback
area(::Shape) = throw(MethodError(area))

area(Rectangle(3, 4))
area(Rectangle(3.0, 4.0))
